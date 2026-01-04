#!/usr/bin/env python3
"""Inspect ERes2NetV2 checkpoint structure."""
import torch

ckpt = torch.load('pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt', map_location='cpu')

print('Checkpoint keys:')
print(list(ckpt.keys())[:30])

print('\n\nSample layer shapes:')
for k in list(ckpt.keys())[:30]:
    if hasattr(ckpt[k], 'shape'):
        print(f'{k}: {ckpt[k].shape}')
    else:
        print(f'{k}: {type(ckpt[k])}')

print('\n\nTotal keys:', len(ckpt.keys()))
