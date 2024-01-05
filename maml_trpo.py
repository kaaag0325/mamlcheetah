import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector

from maml_rl.metalearners.base import GradientBasedMetaLearner

# from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.utils.torch_utils import (
    detach_distribution,
    to_numpy,
    vector_to_parameters,
    weighted_mean,
)


class MAMLTRPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].

    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.

    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """

    def __init__(self, policy, fast_lr=0.5, first_order=False, device="cpu"):
        """
        __init__ 함수는 MAMLTRPO 클래스를 초기화하는 함수이다.

        Args:
            policy (maml_rl.policies.Policy): Policy network (torch.nn.Module)
            fast_lr (float, optional): fast adaptation step-size. Defaults to 0.5.
            first_order (bool, optional): first order approximation. Defaults to False.
            device (str, optional): cpu or cuda. Defaults to "cpu".
        """

        super().__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    async def adapt(
        self,
        episodes,  # train_futures
        first_order=None,
    ):
        """
        adapt 함수는 episodes를 받아서 policy를 업데이트하는 함수이다.
        episodes는 train_futures(multi-task sampler)를 통해 얻은 episodes이다.
        episodes는 (observations, actions, rewards, lengths)로 구성되어있다.

        Args:
            episodes (tuple): (observations, actions, rewards, lengths)
            first_order (bool, optional): first order approximation. Defaults to None.

        Returns:
            params (torch.Tensor): policy parameters after adaptation
        """

        if first_order is None:
            first_order = self.first_order

        # Loop over the number of steps of adaptation
        params = None
        for futures in episodes:
            inner_loss = reinforce_loss(self.policy, await futures, params=params)
            params = self.policy.update_params(
                inner_loss,
                params=params,
                step_size=self.fast_lr,
                first_order=first_order,
            )

        return params

    def hessian_vector_product(self, kl, damping=1e-2):
        """
        hessian_vector_product 함수는 kl divergence를 받아서 hessian vector product(2차 도함수)를 계산하는 함수이다.
        conjugate_gradient 함수에서 사용된다.

        Args:
            kl (torch.Tensor): kl divergence
            damping (float, optional): damping coefficient. Defaults to 1e-2.
                damping coefficient는 hessian matrix의 대각 성분에 더해진다.
                (hessian matrix의 대각 성분이 0이면 역행렬이 존재하지 않기 때문에 이를 방지하기 위해 사용된다.)
                (역행렬이 존재하지 않으면 conjugate_gradient (hessian matrix의 역행렬을 구하는 과정)를 수행할 수 없다.)

        Returns:
            _product (function): hessian vector product(2차 도함수)
        """

        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)  # type: ignore
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(
                grad_kl_v, self.policy.parameters(), retain_graph=retain_graph  # type: ignore
            )
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        """
        surrogate loss는 policy(π)를 업데이트하기 위한 loss이다.
        TRPO, PPO, natural policy gradient 등에서 사용된다.
       

        surrogate loss는 다음과 같이 정의된다.
        L(π) = E_π[log(π(a|s)/π_old(a|s)) * A(s, a)]

        위 식을 간단하게 해석하자면
        기존 policy(π_old)와 현재 policy(π)의 분포의 차이를 최소화하면서 advantage function(A(s, a))을 최대화하는 policy(π)를 찾는다.
        기대값(E_π)과 로그 부분은 kl divergence를 의미한다.
        (KLD가 작은 것이 분포의 차이가 작다는 것을 의미하고, policy가 업데이트될 때 큰 변화를 주지 않는다는 것을 의미한다.)

        A(s,a) is the advantage of taking action a in state s, Q(s,a) is the expected reward of taking action a in state s,
        and V(s) is the expected reward of following the current policy in state s.

        A(s, a)는 advantage function으로, 다음과 같이 정의된다.
        A(s, a) = Q(s, a) - V(s)
        Q(s, a)는 state-action value function이고 V(s)는 state value function이다.
        advantage function은 policy(π)를 업데이트하기 위한 방향성을 제공한다.
        (advantage function이 크면 policy를 업데이트할 때 큰 변화를 준다.)

        π_old는 π를 업데이트하기 전의 π이다.
        π_old를 사용하는 이유는 π_old를 사용하지 않으면 π가 너무 큰 변화를 일으킬 수 있기 때문이다.
        (π_old를 사용하면 π가 π_old와 비슷한 분포를 가지도록 업데이트된다.)
        (이 때 비슷한 분포를 가지도록 업데이트된다는 것은 π_old와 π의 kl divergence가 작아진다는 것을 의미한다.)

        Args:
            train_futures (tuple): MultiTaskSampler를 통해 얻은 train episodes
            valid_futures (tuple): MultiTaskSampler를 통해 얻은 validation episodes
            old_pi (torch.Tensor, optional): π_old. Defaults to None.

        Returns:
            losses.mean() (torch.Tensor): surrogate loss
            kls.mean() (torch.Tensor): kl divergence
            old_pi (torch.Tensor): π_old
        """

        first_order = (old_pi is not None) or self.first_order
        params = await self.adapt(train_futures, first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = await valid_futures
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = pi.log_prob(valid_episodes.actions) - old_pi.log_prob(
                valid_episodes.actions
            )
            ratio = torch.exp(log_ratio)

            losses = -weighted_mean(
                ratio * valid_episodes.advantages, lengths=valid_episodes.lengths
            )
            kls = weighted_mean(kl_divergence(pi, old_pi), lengths=valid_episodes.lengths)

        return losses.mean(), kls.mean(), old_pi

    def step(
        self,
        train_episodes,  # train_futures (super class의 변수와 맞추기 위해 변경함)
        valid_episodes,  # valid_futures (super class의 변수와 맞추기 위해 변경함)
        max_kl=1e-3,
        cg_iters=10,
        cg_damping=1e-2,
        ls_max_steps=10,
        ls_backtrack_ratio=0.5,
        vanilla_trpo=False,  # Use Vanilla TRPO version if True, MAMLTRPO version if False
    ):
        """
        step 함수는 policy를 업데이트하는 함수이다.

        1. MAML TRPO version

        계산한 Surrogate losses들의 평균(기대값)을 구하여 policy를 업데이트한다.
        (num_tasks로 나누는 부분이 이에 해당한다.)

        2. Vanilla TRPO version
        (주관적 해석이 포함되어있음)

        기존 TRPO에서는 Single Task로 policy를 업데이트한다.
        MAML TRPO에서는 MultiTaskSampler를 통해 얻은 episodes들을 사용하여 policy를 업데이트한다.

        Multi-Task RL과는 다르지만, 코드 전체적인 구조를 변경하지 않기 위해
        Multi-Task의 결과 중 가장 작은 loss를 가진 episode를 선택하여 policy parameter를 업데이트한다.
        (Multi-Task RL은 다른 Task를 동시에 학습하는 것이고, MAML TRPO는 단일 Task를 여러개의 episode로 진행하는
        것이기에 여기에 사용된 Multi-Task는 Multi-Task RL과는 다르다.)

        이는 다르게 해석하면 Multi-Task의 episodes들이 같은 Task에서 Exploration을 한 것이라고 볼 수 있을 것이다.

        Returns:
            logs (dict): 업데이트 이전/이후의 surrogate loss와 kl divergence
        """

        num_tasks = len(train_episodes[0])
        logs = {}

        # 1. Compute the surrogate loss
        old_losses, old_kls, old_pis = self._async_gather(
            [
                self.surrogate_loss(train, valid, old_pi=None)
                for (train, valid) in zip(zip(*train_episodes), valid_episodes)
            ]
        )

        logs["loss_before"] = to_numpy(old_losses)
        logs["kl_before"] = to_numpy(old_kls)

        if vanilla_trpo:
            old_loss = min(old_losses)  # Vanilla TRPO version
        else:
            old_loss = sum(old_losses) / num_tasks  # MAML TRPO version

        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)  # type: ignore
        grads = parameters_to_vector(grads)

        # 2. Compute the step direction with Conjugate Gradient
        if vanilla_trpo:
            old_kl = min(old_kls)  # Vanilla TRPO version
        else:
            old_kl = sum(old_kls) / num_tasks  # MAML TRPO version

        hessian_vector_product = self.hessian_vector_product(old_kl, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # 3. Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # 4. Line search to update the parameters
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters())

            losses, kls, _ = self._async_gather(
                [
                    self.surrogate_loss(train, valid, old_pi=old_pi)
                    for (train, valid, old_pi) in zip(zip(*train_episodes), valid_episodes, old_pis)
                ]
            )

            if vanilla_trpo:
                improve = min(losses) - old_loss  # Vanilla TRPO version
                kl = min(kls)  # Vanilla TRPO version
            else:
                improve = (sum(losses) / num_tasks) - old_loss  # MAML TRPO version
                kl = sum(kls) / num_tasks  # MAML TRPO version

            if (improve.item() < 0.0) and (kl.item() < max_kl):  # type: ignore
                logs["loss_after"] = to_numpy(losses)
                logs["kl_after"] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

        return logs
