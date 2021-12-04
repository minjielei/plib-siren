import torch
import diff_operators


def image_mse(model_output, gt, weight=1.):
    return {'img_loss': (weight * (model_output['model_out'] - gt['coords']) ** 2).mean()}

def image_mse_avg(model_output, gt, weight=1.):
    return {'img_loss': (torch.mean(weight, 0) * (torch.mean(model_output['model_out'], 0) - torch.mean(gt['coords'], 0)) ** 2).mean()}

def image_l1(model_output, gt, weight=1.):
    return {'img_loss': (weight * torch.abs(model_output['model_out'] - gt['coords'])).mean()}


def image_mse_TV_prior(k1, model, model_output, gt, weight=1.):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    return {'img_loss': (weight * (model_output['model_out'] - gt['coords']) ** 2).mean(),
            'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                rand_output['model_out'], rand_output['model_in']))).mean()}


def image_mse_FH_prior(k1, model, model_output, gt, weight=1.):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(rand_output['model_out'],
                                                 rand_output['model_in'])
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    return {'img_loss': (weight * (model_output['model_out'] - gt['coords']) ** 2).mean(),
            'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}