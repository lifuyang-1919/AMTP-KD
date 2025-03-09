import torch
from torch.nn.utils import clip_grad_norm_

from pcdet.config import cfg
from pcdet.utils import common_utils
from pcdet.models.dense_heads import CenterHead, AnchorHeadTemplate


def adjust_batch_info_teacher(batch):
    if cfg.KD.get('DIFF_VOXEL', None):
        batch['voxels_stu'] = batch.pop('voxels')
        batch['voxel_coords_stu'] = batch.pop('voxel_coords')
        batch['voxel_num_points_stu'] = batch.pop('voxel_num_points')

        batch['voxels'] = batch.pop('voxels_tea')
        batch['voxel_coords'] = batch.pop('voxel_coords_tea')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_tea')

    teacher_pred_flag = False
    teacher_target_dict_flag = False
    teacher_decoded_pred_flag = False

    # LOGIT KD
    if cfg.KD.get('LOGIT_KD', None) and cfg.KD.LOGIT_KD.ENABLED:
        if cfg.KD.LOGIT_KD.MODE in ['raw_pred', 'target']:
            teacher_pred_flag = True
            teacher_target_dict_flag = True
        elif cfg.KD.LOGIT_KD.MODE == 'decoded_boxes':
            teacher_decoded_pred_flag = True
        else:
            raise NotImplementedError

    if cfg.KD.get('LABEL_ASSIGN_KD', None) and cfg.KD.LABEL_ASSIGN_KD.ENABLED:
        teacher_decoded_pred_flag = True
    
    if cfg.KD.get('MASK', None):
        if cfg.KD.MASK.get('FG_MASK', None):
            teacher_target_dict_flag = True
        
        if cfg.KD.MASK.get('BOX_MASK', None):
            teacher_decoded_pred_flag = True
        
        if cfg.KD.MASK.get('SCORE_MASK', None):
            teacher_pred_flag = True

    batch['teacher_pred_flag'] = teacher_pred_flag
    batch['teacher_target_dict_flag'] = teacher_target_dict_flag
    batch['teacher_decoded_pred_flag'] = teacher_decoded_pred_flag


def adjust_batch_info_student(batch):
    if cfg.KD.get('DIFF_VOXEL', None):
        del batch['voxels']
        del batch['voxel_coords']
        del batch['voxel_num_points']

        batch['voxels'] = batch.pop('voxels_stu')
        batch['voxel_coords'] = batch.pop('voxel_coords_stu')
        batch['voxel_num_points'] = batch.pop('voxel_num_points_stu')


def add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=None):
    if cfg.KD.get('FEATURE_KD', None) and cfg.KD.FEATURE_KD.ENABLED:
        feature_name = cfg.KD.FEATURE_KD.get('FEATURE_NAME_TEA', cfg.KD.FEATURE_KD.FEATURE_NAME)
        batch[feature_name + '_tea'] = batch[feature_name].detach()

    if cfg.KD.get('PILLAR_KD', None) and cfg.KD.PILLAR_KD.ENABLED:
        feature_name_tea = cfg.KD.PILLAR_KD.FEATURE_NAME_TEA
        batch['voxel_features_tea'] = batch.pop(feature_name_tea)

    if cfg.KD.get('VFE_KD', None) and cfg.KD.VFE_KD.ENABLED:
        batch['point_features_tea'] = batch.pop('point_features')
        batch['pred_tea'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
        if cfg.KD.VFE_KD.get('SAVE_INDS', None):
            batch['unq_inv_pfn_tea'] = batch.pop('unq_inv_pfn')
        if cfg.KD.VFE_KD.get('SAVE_3D_FEAT', None):
            batch['spatial_features_tea'] = batch.pop('spatial_features')

    if cfg.KD.get('ROI_KD', None) and cfg.KD.ROI_KD.ENABLED:
        batch['rcnn_cls_tea'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_cls')
        batch['rcnn_reg_tea'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_reg')
        batch['roi_head_target_dict_tea'] = teacher_model.roi_head.forward_ret_dict

    if cfg.KD.get('SAVE_COORD_TEA', None):
        batch['voxel_coords_tea'] = batch.pop('voxel_coords')
    
    if batch.get('teacher_target_dict_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            batch['target_dicts_tea'] = teacher_model.dense_head.forward_ret_dict['target_dicts']
            # import pdb;pdb.set_trace()
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch['spatial_mask_tea'] = batch['spatial_features'].sum(dim=1) != 0

    if batch.get('teacher_pred_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            batch['pred_tea'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
            # import pdb;pdb.set_trace()
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch['cls_preds_tea'] = teacher_model.dense_head.forward_ret_dict['cls_preds']
            batch['box_preds_tea'] = teacher_model.dense_head.forward_ret_dict['box_preds']
            batch['dir_cls_preds_tea'] = teacher_model.dense_head.forward_ret_dict['dir_cls_preds']
            # import pdb;pdb.set_trace()

    if batch.get('teacher_decoded_pred_flag', None):
        if (not teacher_model.training) and teacher_model.roi_head is not None:
            batch['decoded_pred_tea'] = pred_dicts
        elif isinstance(teacher_model.dense_head, CenterHead):
            batch['decoded_pred_tea'] = teacher_model.dense_head.forward_ret_dict['decoded_pred_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch['decoded_pred_tea'] = pred_dicts

def add_teacher_pred_to_batch_2(teacher_model_2, batch, pred_dicts=None):
    if cfg.KD.get('FEATURE_KD', None) and cfg.KD.FEATURE_KD.ENABLED:
        feature_name = cfg.KD.FEATURE_KD.get('FEATURE_NAME_TEA', cfg.KD.FEATURE_KD.FEATURE_NAME)
        batch[feature_name + '_tea2'] = batch[feature_name].detach()

    if cfg.KD.get('PILLAR_KD', None) and cfg.KD.PILLAR_KD.ENABLED:
        feature_name_tea = cfg.KD.PILLAR_KD.FEATURE_NAME_TEA
        batch['voxel_features_tea2'] = batch.pop(feature_name_tea)

    if cfg.KD.get('VFE_KD', None) and cfg.KD.VFE_KD.ENABLED:
        batch['point_features_tea2'] = batch.pop('point_features')
        batch['pred_tea2'] = teacher_model_2.dense_head.forward_ret_dict['pred_dicts']
        if cfg.KD.VFE_KD.get('SAVE_INDS', None):
            batch['unq_inv_pfn_tea2'] = batch.pop('unq_inv_pfn')
        if cfg.KD.VFE_KD.get('SAVE_3D_FEAT', None):
            batch['spatial_features_tea2'] = batch.pop('spatial_features')

    if cfg.KD.get('ROI_KD', None) and cfg.KD.ROI_KD.ENABLED:
        batch['rcnn_cls_tea2'] = teacher_model_2.roi_head.forward_ret_dict.pop('rcnn_cls')
        batch['rcnn_reg_tea2'] = teacher_model_2.roi_head.forward_ret_dict.pop('rcnn_reg')
        batch['roi_head_target_dict_tea2'] = teacher_model_2.roi_head.forward_ret_dict

    if cfg.KD.get('SAVE_COORD_TEA', None):
        batch['voxel_coords_tea2'] = batch.pop('voxel_coords')

    if batch.get('teacher_target_dict_flag', None):
        if isinstance(teacher_model_2.dense_head, CenterHead):
            batch['target_dicts_tea2'] = teacher_model_2.dense_head.forward_ret_dict['target_dicts']
            # import pdb;pdb.set_trace()
        elif isinstance(teacher_model_2.dense_head, AnchorHeadTemplate):
            batch['spatial_mask_tea2'] = batch['spatial_features'].sum(dim=1) != 0

    if batch.get('teacher_pred_flag', None):
        if isinstance(teacher_model_2.dense_head, CenterHead):
            batch['pred_tea2'] = teacher_model_2.dense_head.forward_ret_dict['pred_dicts']
            # import pdb;pdb.set_trace()
        elif isinstance(teacher_model_2.dense_head, AnchorHeadTemplate):
            batch['cls_preds_tea2'] = teacher_model_2.dense_head.forward_ret_dict['cls_preds']
            batch['box_preds_tea2'] = teacher_model_2.dense_head.forward_ret_dict['box_preds']
            batch['dir_cls_preds_tea2'] = teacher_model_2.dense_head.forward_ret_dict['dir_cls_preds']
            # import pdb;pdb.set_trace()

    if batch.get('teacher_decoded_pred_flag', None):
        if (not teacher_model_2.training) and teacher_model_2.roi_head is not None:
            batch['decoded_pred_tea2'] = pred_dicts
        elif isinstance(teacher_model_2.dense_head, CenterHead):
            batch['decoded_pred_tea2'] = teacher_model_2.dense_head.forward_ret_dict['decoded_pred_dicts']
        elif isinstance(teacher_model_2.dense_head, AnchorHeadTemplate):
            batch['decoded_pred_tea2'] = pred_dicts

def add_teacher_pred_to_batch_n(teacher_model, batch, pred_dicts=None, teacher_num=None):
    """
    将第n个teacher的预测添加到batch中
    Args:
        teacher_model: 当前teacher模型
        batch: batch数据
        pred_dicts: 预测字典
        teacher_num: 2-5,表示第几个teacher
    """
    # 设置后缀，用于区分不同teacher的特征
    suffix = f'tea{teacher_num}'

    # 特征知识蒸馏
    if (cfg.KD.get('FEATURE_KD', None) and cfg.KD.FEATURE_KD.ENABLED):
        feature_name = cfg.KD.FEATURE_KD.get('FEATURE_NAME_TEA', cfg.KD.FEATURE_KD.FEATURE_NAME)
        batch[f'{feature_name}_{suffix}'] = batch[feature_name].detach()

    # Pillar知识蒸馏
    if cfg.KD.get('PILLAR_KD', None) and cfg.KD.PILLAR_KD.ENABLED:
        feature_name_tea = cfg.KD.PILLAR_KD.FEATURE_NAME_TEA
        batch[f'voxel_features_{suffix}'] = batch.pop(feature_name_tea)

    # VFE知识蒸馏
    if cfg.KD.get('VFE_KD', None) and cfg.KD.VFE_KD.ENABLED:
        batch[f'point_features_{suffix}'] = batch.pop('point_features')
        batch[f'pred_{suffix}'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
        if cfg.KD.VFE_KD.get('SAVE_INDS', None):
            batch[f'unq_inv_pfn_{suffix}'] = batch.pop('unq_inv_pfn')
        if cfg.KD.VFE_KD.get('SAVE_3D_FEAT', None):
            batch[f'spatial_features_{suffix}'] = batch.pop('spatial_features')

    # ROI知识蒸馏
    if cfg.KD.get('ROI_KD', None) and cfg.KD.ROI_KD.ENABLED:
        batch[f'rcnn_cls_{suffix}'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_cls')
        batch[f'rcnn_reg_{suffix}'] = teacher_model.roi_head.forward_ret_dict.pop('rcnn_reg')
        batch[f'roi_head_target_dict_{suffix}'] = teacher_model.roi_head.forward_ret_dict

    # 保存教师模型坐标信息
    if cfg.KD.get('SAVE_COORD_TEA', None):
        batch[f'voxel_coords_{suffix}'] = batch.pop('voxel_coords')

    # 处理教师模型的目标字典
    if batch.get('teacher_target_dict_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            batch[f'target_dicts_{suffix}'] = teacher_model.dense_head.forward_ret_dict['target_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch[f'spatial_mask_{suffix}'] = batch['spatial_features'].sum(dim=1) != 0

    # 处理教师模型的预测结果
    if batch.get('teacher_pred_flag', None):
        if isinstance(teacher_model.dense_head, CenterHead):
            batch[f'pred_{suffix}'] = teacher_model.dense_head.forward_ret_dict['pred_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch[f'cls_preds_{suffix}'] = teacher_model.dense_head.forward_ret_dict['cls_preds']
            batch[f'box_preds_{suffix}'] = teacher_model.dense_head.forward_ret_dict['box_preds']
            batch[f'dir_cls_preds_{suffix}'] = teacher_model.dense_head.forward_ret_dict['dir_cls_preds']

    # 处理教师模型的解码后预测结果
    if batch.get('teacher_decoded_pred_flag', None):
        if (not teacher_model.training) and teacher_model.roi_head is not None:
            batch[f'decoded_pred_{suffix}'] = pred_dicts
        elif isinstance(teacher_model.dense_head, CenterHead):
            batch[f'decoded_pred_{suffix}'] = teacher_model.dense_head.forward_ret_dict['decoded_pred_dicts']
        elif isinstance(teacher_model.dense_head, AnchorHeadTemplate):
            batch[f'decoded_pred_{suffix}'] = pred_dicts
def get_temperature_from_lr(optimizer, min_temp=2, max_temp=200.0):  #lr-softmax （2，200）
    """
    Calculate temperature T based on current learning rate.
    Args:
        optimizer: The optimizer containing learning rate info
        min_temp: Minimum temperature value (default: 0.1)
        max_temp: Maximum temperature value (default: 10.0)
    Returns:
        float: Temperature value T
    """
    if not hasattr(optimizer, 'param_groups'):
        return 1.0  # Default temperature if no learning rate info

    # import pdb;pdb.set_trace()
    # current_lr = optimizer.param_groups[0]['lr']
    # current_lr2 = optimizer.param_groups[1]['lr']
    # # initial_lr = optimizer.param_groups[0].get('initial_lr', 0.003)
    current_lr = optimizer.lr
    initial_lr = optimizer.param_groups[0].get('initial_lr', 0.003)
    # print('current_lr',current_lr)
    # print('current_lr1', current_lr1)
    # print('current_lr2', current_lr2)
    # print('initial_lr',initial_lr)

    # Calculate temperature based on ratio of current_lr to initial_lr
    # As lr decreases, temperature will decrease (making softmax sharper)
    ratio = current_lr / initial_lr
    if min_temp==max_temp and max_temp==0:
        temperature = 200.0 if ratio >= 0.01 else (2.0 if ratio >= 0.005 else 0.0)
    else:
        temperature = max_temp - (max_temp - min_temp) * ratio
    # temperature = max_temp - (max_temp - min_temp) * ratio
    return temperature

def forward(model, teacher_model, batch, optimizer, extra_optim, optim_cfg, load_data_to_gpu,
            teacher_model_2=None, teacher_num=None, teacher_models=None, **kwargs):
    optimizer.zero_grad()
    if extra_optim is not None:
        extra_optim.zero_grad()
    # lr-softmax
    # if optim_cfg.OPTIMIZER == 'adam_onecycle':
    #     temperature = get_temperature_from_lr(optimizer, optim_cfg.T_range[0], optim_cfg.T_range[1])  #T0=2，T1=200
    #     batch['temperature'] = temperature

    with torch.no_grad():
        adjust_batch_info_teacher(batch)
        load_data_to_gpu(batch)
        if teacher_model.training:
            batch = teacher_model(batch)
            pred_dicts = None
            add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=pred_dicts)
            if teacher_num == 2:
                pred_dicts_2 = None
                batch = teacher_model_2(batch)
                add_teacher_pred_to_batch_2(teacher_model_2, batch, pred_dicts=pred_dicts_2)
        else:
            pred_dicts, ret_dict = teacher_model(batch)
            add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=pred_dicts)
            if teacher_num == 2:
                pred_dicts_2, ret_dict_2 = teacher_model_2(batch)
                add_teacher_pred_to_batch_2(teacher_model_2, batch, pred_dicts=pred_dicts_2)
        # 处理teacher 2-5
        if teacher_models is not None:  # ----------- tea_5 ---------------
            for i, teacher in enumerate(teacher_models, start=2):  # 从2开始编号
                if teacher is not None:
                    if teacher.training:
                        batch = teacher(batch)
                        add_teacher_pred_to_batch_n(teacher, batch, pred_dicts=None, teacher_num=i)
                    else:
                        curr_pred_dicts, curr_ret_dict = teacher(batch)
                        add_teacher_pred_to_batch_n(teacher, batch, pred_dicts=curr_pred_dicts, teacher_num=i)

        # add_teacher_pred_to_batch(teacher_model, batch, pred_dicts=pred_dicts)
        # if teacher_num is not None:
        #     add_teacher_pred_to_batch_2(teacher_model_2, batch, pred_dicts=pred_dicts_2)

    adjust_batch_info_student(batch)

    ret_dict, tb_dict, disp_dict = model(batch)
    loss = ret_dict['loss'].mean()

    loss.backward()
    clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

    optimizer.step()
    if extra_optim is not None:
        extra_optim.step()

    return loss, tb_dict, disp_dict
