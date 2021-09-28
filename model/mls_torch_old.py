import torch
from torch._C import device
import torch.nn as nn
import numpy as np

class MotionEstimation(nn.Module):
    def __init__(self):
        super(MotionEstimation,self).__init__()
    def forward(self,sp, dp):
        b, _, _= sp.shape
        print(sp.shape)
        b_tensor = None
        m_tensor = None
        w_tensor = None
        phat_tensor = None
        qhat_tensor = None
        for i in range(b):
            gridX = torch.arange(32, dtype=torch.float32).cuda()
            gridY = torch.arange(32, dtype=torch.float32).cuda()
            vy, vx = torch.meshgrid(gridX, gridY)
            # print(vy.device,vx.device)
            affine, M, W, phat, qhat = self.mls_affine_deformation(
                vy, vx, sp[i], dp[i])
            # if b_tensor is None:
            #     b_tensor=affine.unsqueeze(0)
            #     m_tensor=M.unsqueeze(0)
            #     w_tensor=W.unsqueeze(0)
            #     phat_tensor=phat.unsqueeze(0)
            #     qhat_tensor=qhat.unsqueeze(0)
            # else:
            #     b_tensor=torch.cat([b_tensor,affine.unsqueeze(0)],dim=0)
            #     m_tensor=torch.cat([m_tensor,M.unsqueeze(0)],dim=0)
            #     w_tensor=torch.cat([w_tensor,W.unsqueeze(0)],dim=0)
            #     phat_tensor=torch.cat([phat_tensor,phat.unsqueeze(0)],dim=0)
            #     qhat_tensor=torch.cat([qhat_tensor,qhat.unsqueeze(0)],dim=0)
            # # print(affine.dtype)
            # # print(b_tensor)
        # return  b_tensor,m_tensor,w_tensor,phat_tensor,qhat_tensor
        # return torch.as_tensor([item.numpy() for item in b_list]), \
        #     torch.as_tensor([item.numpy() for item in m_list]),\
        #     torch.as_tensor([item.numpy() for item in w_list]), \
        #     torch.as_tensor([item.numpy() for item in phat_list]),\
        #     torch.as_tensor([item.numpy() for item in b_list])
    def mls_affine_deformation(self,vy, vx, p, q, alpha=1.0, eps=1e-8):
        # Change (x, y) to (row, col)
        # q = torch.ascontiguousarray(q[:, [1, 0]].astype(torch.int16))
        # p = torch.ascontiguousarray(p[:, [1, 0]].astype(torch.int16)
        print(p.shape)
        q = q[:, [1, 0]].to(torch.int16)
        p = p[:, [1, 0]].to(torch.int16)
        # print("q",q.shape,"p",p.shape)
        # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
        p, q = q, p

        grow = vx.shape[0]  # grid rows
        gcol = vx.shape[1]  # grid cols
        ctrls = p.shape[0]  # control points
        # print(grow,gcol,ctrls)
        # Precompute
        # [ctrls, 2, 1, 1]
        reshaped_p = p.reshape(ctrls, 2, 1, 1)
        reshaped_v = torch.vstack(
            (vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
        # print("shape:",reshaped_v.shape)
        w = 1.0 / (torch.sum((reshaped_p - reshaped_v).cuda().to(torch.float32)
                ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
        # [ctrls, grow, gcol]
        w /= torch.sum(w, axis=0, keepdims=True)
        # print(grow,gcol)
        pstar = torch.zeros((2, grow, gcol)).cuda()
        for i in range(ctrls):
            # [2, grow, gcol]
            pstar += w[i] * reshaped_p[i]

        # [ctrls, 2, grow, gcol]
        phat = reshaped_p - pstar
        # [ctrls, 2, 1, grow, gcol]
        phat = phat.reshape(ctrls, 2, 1, grow, gcol)
        # [ctrls, 1, 2, grow, gcol]
        phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)
        # [ctrls, 1, 1, grow, gcol]
        reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)
        pTwp = torch.zeros((2, 2, grow, gcol)).cuda()
        for i in range(ctrls):
            pTwp += phat[i] * reshaped_w[i] * phat1[i]
        del phat1
        # try:
        # [grow, gcol, 2, 2]
        print(pTwp.shape)
        inv_pTwp = torch.linalg.inv(pTwp.permute(2, 3, 0, 1))

        # [2, grow, gcol]
        mul_left = reshaped_v - pstar
        reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).permute(
            2, 3, 0, 1)        # [grow, gcol, 1, 2]
        # [ctrls, 2, 1, grow, gcol]
        mul_right = torch.multiply(reshaped_w, phat, out=phat)
        # [ctrls, grow, gcol, 2, 1]
        reshaped_mul_right = mul_right.permute(0, 3, 4, 1, 2)
        # [ctrls, grow, gcol, 1, 1]
        out_A = mul_right.reshape(2, ctrls, grow, gcol, 1, 1)[0]
        A = torch.matmul(torch.matmul(reshaped_mul_left, inv_pTwp),
                        reshaped_mul_right)    # [ctrls, grow, gcol, 1, 1]
        # [ctrls, 1, grow, gcol]
        A = A.reshape(ctrls, 1, grow, gcol)
        del mul_right, reshaped_mul_right

        # Calculate q
        # [ctrls, 2, 1, 1]
        reshaped_q = q.reshape((ctrls, 2, 1, 1))
        qstar = torch.zeros((2, grow, gcol)).cuda()
        for i in range(ctrls):
            # [2, grow, gcol]
            qstar += w[i] * reshaped_q[i]
        # del w, reshaped_w

        # Get final image transfomer -- 3-D array
        transformers = torch.zeros((2, grow, gcol)).cuda()
        for i in range(ctrls):
            transformers += A[i] * (reshaped_q[i] - qstar)
        transformers += qstar
        del A

        qhat = reshaped_q-qstar
        qhat = qhat.reshape(ctrls, 2, 1, grow, gcol)
        # Removed the points outside the border
        transformers[transformers < 0] = 0
        transformers[0][transformers[0] > grow - 1] = 0
        transformers[1][transformers[1] > gcol - 1] = 0

        # print(transformers.shape)
        # print(transformers)
        # print(transformers.dtype)
        return transformers, pTwp, w, phat, qhat  # .to(torch.int32)


def mls_similarity_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """ Similarity deformation

    Parameters
    ----------
    vx, vy: ndarray
        coordinate grid, generated by torch.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points
    q: ndarray
        an array with size [n, 2], final control points
    alpha: float
        parameter used by weights
    eps: float
        epsilon

    Return
    ------
        A deformed image.
    """
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]].to(torch.int16)
    p = p[:, [1, 0]].to(torch.int16)
    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    # [ctrls, 2, 1, 1]
    reshaped_p = p.reshape(ctrls, 2, 1, 1)
    reshaped_v = torch.vstack(
        (vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / (torch.sum((reshaped_p - reshaped_v).to(torch.float32)
               ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
    # [ctrls, grow, gcol]
    w /= torch.sum(w, axis=0, keepdims=True)

    pstar = torch.zeros((2, grow, gcol))
    for i in range(ctrls):
        # [2, grow, gcol]
        pstar += w[i] * reshaped_p[i]

    # [ctrls, 2, grow, gcol]
    phat = reshaped_p - pstar
    # [ctrls, 1, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)
    # [ctrls, 1, 1, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)

    mu = torch.zeros((grow, gcol))
    for i in range(ctrls):
        mu += w[i] * (phat[i] ** 2).sum(0)
    # [1, grow, gcol]
    reshaped_mu = mu.reshape(1, grow, gcol)

    # [2, grow, gcol]
    vpstar = reshaped_v - pstar
    # [2, 1, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)
    # [2, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(
        2, 1, grow, gcol)              # [2, 1, grow, gcol]
    # [2, 2, grow, gcol]
    mul_right = torch.cat((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)

    # Calculate q
    # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))
    qstar = torch.zeros((2, grow, gcol))
    for i in range(ctrls):
        # [2, grow, gcol]
        qstar += w[i] * reshaped_q[i]

    # Get final image transfomer -- 3-D array
    temp = torch.zeros((grow, gcol, 2))
    for i in range(ctrls):
        # [2, grow, gcol]
        neg_phat_verti = phat[i, [1, 0]]
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(
            1, 2, grow, gcol)              # [1, 2, grow, gcol]
        # [2, 2, grow, gcol]
        mul_left = torch.cat(
            (reshaped_phat[i], reshaped_neg_phat_verti), axis=0)

        A = torch.matmul((reshaped_w[i] * mul_left).permute(2, 3, 0, 1),
                         mul_right.permute(2, 3, 0, 1))                                  # [grow, gcol, 2, 2]

        # [2, grow, gcol]
        qhat = reshaped_q[i] - qstar
        reshaped_qhat = qhat.reshape(1, 2, grow, gcol).permute(
            2, 3, 0, 1)            # [grow, gcol, 1, 2]

        # Get final image transfomer -- 3-D array
        # [grow, gcol, 2]
        temp += torch.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)

    # [2, grow, gcol]
    transformers = temp.permute(2, 0, 1) / reshaped_mu + qstar

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers


def mls_rigid_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """ Rigid deformation

    Parameters
    ----------
    vx, vy: ndarray
        coordinate grid, generated by torch.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points
    q: ndarray
        an array with size [n, 2], final control points
    alpha: float
        parameter used by weights
    eps: float
        epsilon

    Return
    ------
        A deformed image.
    """
    # Change (x, y) to (row, col)

    q = q[:, [1, 0]].to(torch.int16)
    p = p[:, [1, 0]].to(torch.int16)
    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    # [ctrls, 2, 1, 1]
    reshaped_p = p.reshape(ctrls, 2, 1, 1)
    reshaped_v = torch.vstack(
        (vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / (torch.sum((reshaped_p - reshaped_v).to(torch.float32)
               ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
    # [ctrls, grow, gcol]
    w /= torch.sum(w, axis=0, keepdims=True)

    pstar = torch.zeros((2, grow, gcol))
    for i in range(ctrls):
        # [2, grow, gcol]
        pstar += w[i] * reshaped_p[i]

    # [2, grow, gcol]
    vpstar = reshaped_v - pstar
    # [2, 1, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)
    # [2, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(
        2, 1, grow, gcol)              # [2, 1, grow, gcol]
    # [2, 2, grow, gcol]
    mul_right = torch.cat((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)
    reshaped_mul_right = mul_right.reshape(
        2, 2, grow, gcol)                            # [2, 2, grow, gcol]

    # Calculate q
    # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))
    qstar = torch.zeros((2, grow, gcol))
    for i in range(ctrls):
        # [2, grow, gcol]
        qstar += w[i] * reshaped_q[i]

    temp = torch.zeros((grow, gcol, 2))
    for i in range(ctrls):
        # [2, grow, gcol]
        phat = reshaped_p[i] - pstar
        # [1, 2, grow, gcol]
        reshaped_phat = phat.reshape(1, 2, grow, gcol)
        # [1, 1, grow, gcol]
        reshaped_w = w[i].reshape(1, 1, grow, gcol)
        # [2, grow, gcol]
        neg_phat_verti = phat[[1, 0]]
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(
            1, 2, grow, gcol)              # [1, 2, grow, gcol]
        # [2, 2, grow, gcol]
        mul_left = torch.cat((reshaped_phat, reshaped_neg_phat_verti), axis=0)

        A = torch.matmul((reshaped_w * mul_left).permute(2, 3, 0, 1),
                         reshaped_mul_right.permute(2, 3, 0, 1))                       # [grow, gcol, 2, 2]

        # [2, grow, gcol]
        qhat = reshaped_q[i] - qstar
        reshaped_qhat = qhat.reshape(1, 2, grow, gcol).permute(
            2, 3, 0, 1)            # [grow, gcol, 1, 2]

        # Get final image transfomer -- 3-D array
        # [grow, gcol, 2]
        temp += torch.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)

    # [2, grow, gcol]
    temp = temp.permute(2, 0, 1)
    # [1, grow, gcol]
    normed_temp = torch.linalg.norm(temp, axis=0, keepdims=True)
    normed_vpstar = torch.linalg.norm(
        vpstar, axis=0, keepdims=True)                       # [1, grow, gcol]
    transformers = temp / normed_temp * normed_vpstar + \
        qstar                          # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers  # .to(torch.device("cuda"))


if __name__ == "__main__":

    import cv2
    import numpy as np
    img = cv2.imread("img/toy.jpg")
    h, w, _ = img.shape
    print(h, w)
    p = torch.tensor([
        [30, 155], [125, 155], [225, 155],
        [100, 235], [160, 235], [85, 295], [180, 293]
    ])
    q = torch.tensor([
        [42, 211], [125, 155], [235, 100],
        [80, 235], [140, 235], [85, 295], [180, 295]
    ])
    gridX = torch.arange(w, dtype=torch.int16)
    gridY = torch.arange(h, dtype=torch.int16)
    vy, vx = torch.meshgrid(gridX, gridY)
    affine = mls_affine_deformation(vy, vx, p, q, alpha=1)
    similar = mls_similarity_deformation(vy, vx, p, q, alpha=1)
    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    print(affine.shape, similar.shape, rigid.shape)
    # cv2.imshow("test",aug1)
    # cv2.waitKey(0)
