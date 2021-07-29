import copy

import numpy as np
import torch

from scores import ContentScore, StyleScore


class StyleTransfer:

    @staticmethod
    def inject_scoring(base, content, style, content_layers, style_layers):
        model = torch.nn.Sequential()
        c_i = 0

        for name, layer in copy.deepcopy(base).named_children():
            if hasattr(layer, "inplace"):
                layer.inplace = False

            model.add_module(name, layer)
            if isinstance(layer, torch.nn.Conv2d):
                c_i += 1
            else:
                continue

            if c_i in content_layers:
                target = model(content).detach()
                model.add_module(f"content_score_{c_i}", ContentScore(target))

            if c_i in style_layers:
                target = model(style).detach()
                model.add_module(f"style_score_{c_i}", StyleScore(target))

        def is_score(layer):
            return isinstance(layer, (ContentScore, StyleScore))

        last_i = max(i for i, l in enumerate(model.children()) if is_score(l))
        model = model[:(last_i + 1)]

        return model

    def __call__(self,
                 base: torch.nn.Module,
                 content: torch.Tensor,
                 style: torch.Tensor,
                 content_layers: set[int],
                 style_layers: set[int],
                 style_weight=1e6):

        x = content.clone()
        model = self.inject_scoring(base, content, style, content_layers, style_layers)
        optimizer = torch.optim.Adam([x.requires_grad_()], lr=0.03)

        content_layers = [l for l in model.children() if isinstance(l, ContentScore)]
        style_layers = [l for l in model.children() if isinstance(l, StyleScore)]

        while True:
            model(x)
            content_loss = torch.stack([l.score for l in content_layers]).mean()
            style_loss = torch.stack([l.score for l in style_layers]).mean()
            loss = style_loss * style_weight + content_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.clamp_(0, 1)
            yield x
