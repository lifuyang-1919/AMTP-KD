import torch
# from torchvision.models import resnet18
import torch_pruning as tp
from pcdet.models.dense_heads.center_head import CenterHead  # head
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone  # backbone 2d
from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarVFE  # vfe
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter
from pcdet.models.backbones_2d.map_to_bev.height_compression import HeightCompression
from pcdet.models.backbones_3d.spconv_backbone import VoxelResBackBone8x, VoxelBackBone8x
from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle
from pcdet.models.dense_heads.anchor_head_multi import AnchorHeadMulti

def sliming_pru(model, batch, Random, prunratio):
    print('orig_model', model)
    example_inputs = batch
    # ---------------------------------------------------
    imp = tp.importance.TaylorImportance()
    if Random:
        imp = tp.importance.RandomImportance()

    ignored_layers = []
    for m in model.modules():

        if isinstance(m, torch.nn.Conv2d) and m.out_channels == 1:
            ignored_layers.append(m)   
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 2:
            ignored_layers.append(m)   
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 3:
            ignored_layers.append(m)   
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 4:
            ignored_layers.append(m)   
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 6:
            ignored_layers.append(m)   
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 8:
            ignored_layers.append(m)   
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 12:
            ignored_layers.append(m)   
        elif isinstance(m, AnchorHeadSingle):
            ignored_layers.append(m)
        elif isinstance(m, AnchorHeadMulti):
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.ConvTranspose2d) and m.out_channels == 256:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 18:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.Conv2d) and m.out_channels == 42:
            ignored_layers.append(m)

        # if isinstance(m, CenterHead):
        #     ignored_layers.append(m)

        if isinstance(m, PillarVFE):
            ignored_layers.append(m)
        if isinstance(m, PointPillarScatter):
            ignored_layers.append(m)
        if isinstance(m, MeanVFE):
            ignored_layers.append(m)
        if isinstance(m, HeightCompression):
            ignored_layers.append(m)
        if isinstance(m, VoxelResBackBone8x):
            ignored_layers.append(m)
        if isinstance(m, VoxelBackBone8x):
            ignored_layers.append(m)

    iterative_steps = 1  # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=prunratio,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print('base_macs:{:.2f}G'.format(base_macs / 1e9))
    print('base_nparams:{:.2f}M'.format(base_nparams / 1e6))
    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.TaylorImportance):
            model.train()
            ret_dict, tb_dict, disp_dict = model(example_inputs)
            loss = ret_dict['loss'].mean()
            loss.backward() # before pruner.step()
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)

        print('base_macs:{:.2f}G'.format(macs / 1e9))
        print('base_nparams:{:.2f}M'.format(nparams / 1e6))


    print('model_pruned', model)
    print('====================================sliming prune end===================================')
    return model