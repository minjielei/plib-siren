import torch
import diff_operators


def image_mse(weight, model_output, gt):
    if weight is None:
        return {'img_loss': ((model_output['model_out'] - gt['coords']) ** 2).mean()}
    else:
        weight = weight[((gt['coords']-1e-6)*len(weight)).long()]
        return {'img_loss': (weight * (model_output['model_out'] - gt['coords']) ** 2).mean()}


def image_l1(weight, model_output, gt):
    if weight is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['coords']).mean()}
    else:
        weight = weight[((gt['coords']-1e-6)*len(weight)).long()]
        return {'img_loss': (weight * torch.abs(model_output['model_out'] - gt['coords'])).mean()}


def image_mse_TV_prior(weight, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    if weight is None:
        return {'img_loss': ((model_output['model_out'] - gt['coords']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}
    else:
        weight = weight[((gt['coords']-1e-6)*len(weight)).long()]
        return {'img_loss': (weight * (model_output['model_out'] - gt['coords']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}


def image_mse_FH_prior(weight, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(rand_output['model_out'],
                                                 rand_output['model_in'])
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    if weight is None:
        weight = weight[((gt['coords']-1e-6)*len(weight)).long()]
        return {'img_loss': ((model_output['model_out'] - gt['coords']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}
    else:
        return {'img_loss': (weight * (model_output['model_out'] - gt['coords']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}


