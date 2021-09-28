

import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def make_coordinate_grid():
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w =32,32
    x = torch.arange(w).type(torch.float32)
    y = torch.arange(h).type(torch.float32)
    # x = x / w
    # y =  y / h  
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze(0), yy.unsqueeze_(0)], 0)
    return meshed.to(device)
def mls_affine_deformation(p, q, alpha=1.0, eps=1e-8):

    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]
    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    reshaped_v=make_coordinate_grid()
    grow = reshaped_v.shape[1]  # grid rows
    gcol = reshaped_v.shape[2]  # grid cols
    ctrls = p.shape[0]  # control points
    # Precompute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    # reshaped_v = torch.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / (torch.sum((reshaped_p - reshaped_v).to(torch.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
    w /= torch.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

    pstar = torch.zeros((2, grow, gcol),device=device)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    phat = phat.reshape(ctrls, 2, 1, grow, gcol)                                        # [ctrls, 2, 1, grow, gcol]
    phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                                       # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwp = torch.zeros((2, 2, grow, gcol),device=device)
    for i in range(ctrls):
        pTwp += phat[i] * reshaped_w[i] * phat1[i]
    del phat1
    # inv_pTwp = torch.linalg.inv(pTwp.permute(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
    inv_pTwp = torch.inverse(pTwp.permute(2, 3, 0, 1))    
    mul_left = reshaped_v - pstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).permute(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = torch.multiply(reshaped_w, phat)                                 # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right = mul_right.permute(0, 3, 4, 1, 2)                             # [ctrls, grow, gcol, 2, 1]
    # out_A = mul_right.reshape(2, ctrls, grow, gcol, 1, 1)[0]                            # [ctrls, grow, gcol, 1, 1]
    A = torch.matmul(torch.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right)#, out=out_A)    # [ctrls, grow, gcol, 1, 1]
    A = A.reshape(ctrls, 1, grow, gcol)                                                 # [ctrls, 1, grow, gcol]
    del mul_right, reshaped_mul_right 
    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = torch.zeros((2, grow, gcol),device=device)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
    qhat=(reshaped_q - qstar)
    qhat = qhat.reshape(ctrls, 2, 1, grow, gcol)
    # loss(pTwp,reshaped_w,phat,qhat)
    del w
    # Get final image transfomer -- 3-D array
    transformers = torch.zeros((2, grow, gcol),device=device)
    for i in range(ctrls):
        transformers += A[i] * (reshaped_q[i] - qstar)
    transformers += qstar
    # del A
    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] >= 1] = 0
    transformers[1][transformers[1] >= 1] = 0
    return transformers,reshaped_w,A,phat,qhat
def mls_rigid_deformation(p, q, alpha=1.0, eps=1e-8):

    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p =p[:, [1, 0]]

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p
    reshaped_v=make_coordinate_grid()
    grow = reshaped_v.shape[1]  # grid rows
    gcol = reshaped_v.shape[2]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    
    w = 1.0 / (torch.sum((reshaped_p - reshaped_v).type(torch.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
    w /= torch.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

    pstar = torch.zeros((2, grow, gcol), device=device)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = torch.cat((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(2, 2, grow, gcol)                            # [2, 2, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = torch.zeros((2, grow, gcol), device=device)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
    
    temp = torch.zeros((grow, gcol, 2),device=device)
    for i in range(ctrls):
        phat = reshaped_p[i] - pstar                                                    # [2, grow, gcol]
        reshaped_phat = phat.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]
        reshaped_w = w[i].reshape(1, 1, grow, gcol)                                     # [1, 1, grow, gcol]
        neg_phat_verti = phat[[1, 0]]                                                   # [2, grow, gcol]
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, grow, gcol)              # [1, 2, grow, gcol]
        mul_left = torch.cat((reshaped_phat, reshaped_neg_phat_verti), axis=0)     # [2, 2, grow, gcol]
        
        A = torch.matmul((reshaped_w * mul_left).transpose(2, 3, 0, 1), 
                        reshaped_mul_right.transpose(2, 3, 0, 1))                       # [grow, gcol, 2, 2]

        qhat = reshaped_q[i] - qstar                                                    # [2, grow, gcol]
        reshaped_qhat = qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)            # [grow, gcol, 1, 2]

        # Get final image transfomer -- 3-D array
        temp += torch.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)                      # [grow, gcol, 2]

    temp = temp.transpose(2, 0, 1)                                                      # [2, grow, gcol]
    normed_temp = torch.linalg.norm(temp, axis=0, keepdims=True)                           # [1, grow, gcol]
    normed_vpstar = torch.linalg.norm(vpstar, axis=0, keepdims=True)                       # [1, grow, gcol]
    transformers = temp / normed_temp * normed_vpstar  + qstar                          # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = -1
    transformers[0][transformers[0] >= 1] = -1
    transformers[1][transformers[1] >= 1] = -1
        
    return transformers,reshaped_w,A,reshaped_phat,reshaped_qhat
def batch_mls(p,q):
    b,num,_=p.shape
    w=torch.ones((b,num,1,1,32,32),device=device)
    tm=torch.ones((b,num,1,32,32),device=device)
    f=torch.ones((b,2,32,32),device=device)
    p_hat=torch.ones((b,num,2,1,32,32),device=device)
    q_hat=torch.ones((b,num,2,1,32,32),device=device)
    for i in range(b):
        temp_f,temp_w,temp_tm,temp_p_hat,temp_qhat=mls_affine_deformation(p[i],q[i])
        f[i]=temp_f
        w[i]=temp_w
        tm[i]=temp_tm
        p_hat[i]=temp_p_hat
        q_hat[i]=temp_qhat
    return f,w,tm,p_hat,q_hat

