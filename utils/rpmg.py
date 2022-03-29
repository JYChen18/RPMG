import torch
import sys
import os 
BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)
import tools

def Rodrigues(w):
    '''
    axis angle -> rotation
    :param w: [b,3]
    :return: R: [b,3,3]
    '''
    w = w.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3)
    b = w.shape[0]
    theta = w.norm(dim=1)
    #print(theta[0])
    #theta = torch.where(t>math.pi/16, torch.Tensor([math.pi/16]).cuda(), t)
    wnorm = w / (w.norm(dim=1,keepdim=True)+0.001)
    #wnorm = torch.nn.functional.normalize(w,dim=1)
    I = torch.eye(3, device=w.get_device()).repeat(b, 1, 1)
    help1 = torch.zeros((b,1,3, 3), device=w.get_device())
    help2 = torch.zeros((b,1,3, 3), device=w.get_device())
    help3 = torch.zeros((b,1,3, 3), device=w.get_device())
    help1[:,:,1, 2] = -1
    help1[:,:,2, 1] = 1
    help2[:,:,0, 2] = 1
    help2[:,:,2, 0] = -1
    help3[:,:,0, 1] = -1
    help3[:,:,1, 0] = 1
    Jwnorm = (torch.cat([help1,help2,help3],1)*wnorm).sum(dim=1)

    return I + torch.sin(theta) * Jwnorm + (1 - torch.cos(theta)) * torch.bmm(Jwnorm, Jwnorm)

logger = 0
def logger_init(ll):
    global logger
    logger = ll
    print('logger init')

class RPMG(torch.autograd.Function):
    '''
    full version. See "simple_RPMG()" for a simplified version.
    Tips:
        1. Use "logger_init()" to initialize the logger, if you want to record some intermidiate variables by tensorboard.
        2. Use sum of L2/geodesic loss instead of mean, since our tau_converge is derivated without considering the scalar introduced by mean loss. 
           See <ModelNet_PC> for an example.
        3. Pass "weight=$YOUR_WEIGHT" instead of directly multiple the weight on rotation loss, if you want to reweight R loss and other losses. 
           See <poselstm-pytorch> for an example.
    '''
    @staticmethod
    def forward(ctx, in_nd, tau, lam, rgt, iter, weight=1):
        proj_kind = in_nd.shape[1]
        if proj_kind == 6:
            r0 = tools.compute_rotation_matrix_from_ortho6d(in_nd)
        elif proj_kind == 9:
            r0 = tools.symmetric_orthogonalization(in_nd)
        elif proj_kind == 4:
            r0 = tools.compute_rotation_matrix_from_quaternion(in_nd)
        elif proj_kind == 10:
            r0 = tools.compute_rotation_matrix_from_10d(in_nd)
        else:
            raise NotImplementedError
        ctx.save_for_backward(in_nd, r0, torch.Tensor([tau,lam, iter, weight]), rgt)
        return r0

    @staticmethod
    def backward(ctx, grad_in):
        in_nd, r0, config,rgt,  = ctx.saved_tensors
        tau = config[0]
        lam = config[1]
        b = r0.shape[0]
        iter = config[2]
        weight = config[3]
        proj_kind = in_nd.shape[1]

        # use Riemannian optimization to get the next goal R
        if tau == -1:
            r_new = rgt
        else:
            # Eucliean gradient -> Riemannian gradient
            Jx = torch.zeros((b, 3, 3)).cuda()
            Jx[:, 2, 1] = 1
            Jx[:, 1, 2] = -1
            Jy = torch.zeros((b, 3, 3)).cuda()
            Jy[:, 0, 2] = 1
            Jy[:, 2, 0] = -1
            Jz = torch.zeros((b, 3, 3)).cuda()
            Jz[:, 0, 1] = -1
            Jz[:, 1, 0] = 1
            gx = (grad_in*torch.bmm(r0, Jx)).reshape(-1,9).sum(dim=1,keepdim=True)
            gy = (grad_in * torch.bmm(r0, Jy)).reshape(-1, 9).sum(dim=1,keepdim=True)
            gz = (grad_in * torch.bmm(r0, Jz)).reshape(-1, 9).sum(dim=1,keepdim=True)
            g = torch.cat([gx,gy,gz],1)

            # take one step
            delta_w = -tau * g

            # update R
            r_new = torch.bmm(r0, Rodrigues(delta_w))

            #this can help you to tune the tau if you don't use L2/geodesic loss.
            if iter % 100 == 0:
                logger.add_scalar('next_goal_angle_mean', delta_w.norm(dim=1).mean(), iter)
                logger.add_scalar('next_goal_angle_max', delta_w.norm(dim=1).max(), iter)
                R0_Rgt = tools.compute_geodesic_distance_from_two_matrices(r0, rgt)
                logger.add_scalar('r0_rgt_angle', R0_Rgt.mean(), iter)
        
        # inverse & project 
        if proj_kind == 6:
            r_proj_1 = (r_new[:, :, 0] * in_nd[:, :3]).sum(dim=1, keepdim=True) * r_new[:, :, 0]
            r_proj_2 = (r_new[:, :, 0] * in_nd[:, 3:]).sum(dim=1, keepdim=True) * r_new[:, :, 0] \
                      + (r_new[:, :, 1] * in_nd[:, 3:]).sum(dim=1, keepdim=True) * r_new[:, :, 1]
            r_reg_1 = lam * (r_proj_1 - r_new[:, :, 0])
            r_reg_2 = lam * (r_proj_2 - r_new[:, :, 1])
            gradient_nd = torch.cat([in_nd[:, :3] - r_proj_1 + r_reg_1, in_nd[:, 3:] - r_proj_2 + r_reg_2], 1)
        elif proj_kind == 9:
            SVD_proj = tools.compute_SVD_nearest_Mnlsew(in_nd.reshape(-1,3,3), r_new)
            gradient_nd = in_nd - SVD_proj + lam * (SVD_proj - r_new.reshape(-1,9))
            R_proj_g = tools.symmetric_orthogonalization(SVD_proj)
            if iter % 100 == 0:
                logger.add_scalar('9d_reflection', (((R_proj_g-r_new).reshape(-1,9).abs().sum(dim=1))>5e-1).sum(), iter)
                logger.add_scalar('reg', (SVD_proj - r_new.reshape(-1, 9)).norm(dim=1).mean(), iter)
                logger.add_scalar('main', (in_nd - SVD_proj).norm(dim=1).mean(), iter)
        elif proj_kind == 4:
            q_1 = tools.compute_quaternions_from_rotation_matrices(r_new)
            q_2 = -q_1
            normalized_nd = tools.normalize_vector(in_nd)
            q_new = torch.where(
                (q_1 - normalized_nd).norm(dim=1, keepdim=True) < (q_2 - normalized_nd).norm(dim=1, keepdim=True),
                q_1, q_2)
            q_proj = (in_nd * q_new).sum(dim=1, keepdim=True) * q_new
            gradient_nd = in_nd - q_proj + lam * (q_proj - q_new)
        elif proj_kind == 10:
            qg = tools.compute_quaternions_from_rotation_matrices(r_new)
            new_x = tools.compute_nearest_10d(in_nd, qg)
            reg_A = torch.eye(4, device=qg.device)[None].repeat(qg.shape[0],1,1) - torch.bmm(qg.unsqueeze(-1), qg.unsqueeze(-2))
            reg_x = tools.convert_A_to_Avec(reg_A)
            gradient_nd = in_nd - new_x + lam * (new_x - reg_x)
            if iter % 100 == 0:
                logger.add_scalar('reg', (new_x - reg_x).norm(dim=1).mean(), iter)
                logger.add_scalar('main', (in_nd - new_x).norm(dim=1).mean(), iter)
                
        return gradient_nd * weight, None, None,None,None,None



class simple_RPMG(torch.autograd.Function):
    '''
    simplified version without tensorboard and r_gt.
    '''
    @staticmethod
    def forward(ctx, in_nd, tau, lam, weight=1):
        proj_kind = in_nd.shape[1]
        if proj_kind == 6:
            r0 = tools.compute_rotation_matrix_from_ortho6d(in_nd)
        elif proj_kind == 9:
            r0 = tools.symmetric_orthogonalization(in_nd)
        elif proj_kind == 4:
            r0 = tools.compute_rotation_matrix_from_quaternion(in_nd)
        elif proj_kind == 10:
            r0 = tools.compute_rotation_matrix_from_10d(in_nd)
        else:
            raise NotImplementedError
        ctx.save_for_backward(in_nd, r0, torch.Tensor([tau,lam, weight]))
        return r0

    @staticmethod
    def backward(ctx, grad_in):
        in_nd, r0, config,  = ctx.saved_tensors
        tau = config[0]
        lam = config[1]
        weight = config[2]
        b = r0.shape[0]
        proj_kind = in_nd.shape[1]

        # use Riemannian optimization to get the next goal R
        # Eucliean gradient -> Riemannian gradient
        Jx = torch.zeros((b, 3, 3)).cuda()
        Jx[:, 2, 1] = 1
        Jx[:, 1, 2] = -1
        Jy = torch.zeros((b, 3, 3)).cuda()
        Jy[:, 0, 2] = 1
        Jy[:, 2, 0] = -1
        Jz = torch.zeros((b, 3, 3)).cuda()
        Jz[:, 0, 1] = -1
        Jz[:, 1, 0] = 1
        gx = (grad_in*torch.bmm(r0, Jx)).reshape(-1,9).sum(dim=1,keepdim=True)
        gy = (grad_in * torch.bmm(r0, Jy)).reshape(-1, 9).sum(dim=1,keepdim=True)
        gz = (grad_in * torch.bmm(r0, Jz)).reshape(-1, 9).sum(dim=1,keepdim=True)
        g = torch.cat([gx,gy,gz],1)

        # take one step
        delta_w = -tau * g

        # update R
        r_new = torch.bmm(r0, Rodrigues(delta_w))

        # inverse & project 
        if proj_kind == 6:
            r_proj_1 = (r_new[:, :, 0] * in_nd[:, :3]).sum(dim=1, keepdim=True) * r_new[:, :, 0]
            r_proj_2 = (r_new[:, :, 0] * in_nd[:, 3:]).sum(dim=1, keepdim=True) * r_new[:, :, 0] \
                      + (r_new[:, :, 1] * in_nd[:, 3:]).sum(dim=1, keepdim=True) * r_new[:, :, 1]
            r_reg_1 = lam * (r_proj_1 - r_new[:, :, 0])
            r_reg_2 = lam * (r_proj_2 - r_new[:, :, 1])
            gradient_nd = torch.cat([in_nd[:, :3] - r_proj_1 + r_reg_1, in_nd[:, 3:] - r_proj_2 + r_reg_2], 1)
        elif proj_kind == 9:
            SVD_proj = tools.compute_SVD_nearest_Mnlsew(in_nd.reshape(-1,3,3), r_new)
            gradient_nd = in_nd - SVD_proj + lam * (SVD_proj - r_new.reshape(-1,9))
        elif proj_kind == 4:
            q_1 = tools.compute_quaternions_from_rotation_matrices(r_new)
            q_2 = -q_1
            normalized_nd = tools.normalize_vector(in_nd)
            q_new = torch.where(
                (q_1 - normalized_nd).norm(dim=1, keepdim=True) < (q_2 - normalized_nd).norm(dim=1, keepdim=True),
                q_1, q_2)
            q_proj = (in_nd * q_new).sum(dim=1, keepdim=True) * q_new
            gradient_nd = in_nd - q_proj + lam * (q_proj - q_new)
        elif proj_kind == 10:
            qg = tools.compute_quaternions_from_rotation_matrices(r_new)
            new_x = tools.compute_nearest_10d(in_nd, qg)
            reg_A = torch.eye(4, device=qg.device)[None].repeat(qg.shape[0],1,1) - torch.bmm(qg.unsqueeze(-1), qg.unsqueeze(-2))
            reg_x = tools.convert_A_to_Avec(reg_A)
            gradient_nd = in_nd - new_x + lam * (new_x - reg_x)
           
        return gradient_nd * weight, None, None,None,None,None
