import numpy as np

import paddle
import torch

import paddle_release
import torch_release

if __name__ == "__main__":
    paddle_wsl_func = paddle_release.WSLDistillerLoss()
    torch_wsl_func = torch_release.WSLDistillerLoss()

    paddle.set_device("cpu")

    for idx in range(100):
        np_s = np.random.rand(32, 1000).astype("float32")
        np_t = np.random.rand(32, 1000).astype("float32")
        np_label = np.random.randint(0, 1000, [32, 1]).astype("int64")

        pd_logits_s = paddle.to_tensor(np_s)
        pd_logits_t = paddle.to_tensor(np_t)
        pd_label = paddle.to_tensor(np_label)
        pd_loss = paddle_wsl_func(pd_logits_s, pd_logits_t,
                                  pd_label).numpy()[0]

        torch_logits_s = torch.tensor(np_s)
        torch_logits_t = torch.tensor(np_t)
        torch_label = torch.tensor(np_label)
        torch_loss = torch_wsl_func(torch_logits_s, torch_logits_t,
                                    torch_label).numpy()
        print("pd loss: {}, torch loss: {}".format(pd_loss, torch_loss))
        assert np.allclose(pd_loss, torch_loss)
