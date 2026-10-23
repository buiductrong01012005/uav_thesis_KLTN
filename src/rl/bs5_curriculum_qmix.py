"""Same Double-Q/Huber/clip-10 QMIX update, with audit diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.rl.bs5_launch_qmix import BS5LaunchQMIX


class BS5CurriculumQMIX(BS5LaunchQMIX):
    def train_batch(self, batch):
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        b = len(batch["reward"])
        q = self.agent(tensors["obs"].view(b * self.n_agents, -1)).view(
            b, self.n_agents, self.n_actions
        )
        chosen = q.gather(2, tensors["actions"].unsqueeze(-1)).squeeze(-1)
        total = self.mixer(chosen, tensors["state"])
        with torch.no_grad():
            online = self.agent(tensors["next_obs"].view(b * self.n_agents, -1)).view(
                b, self.n_agents, self.n_actions
            )
            actions = online.masked_fill(~tensors["next_action_mask"], -1e9).argmax(
                -1, keepdim=True
            )
            target = self.target_agent(tensors["next_obs"].view(b * self.n_agents, -1)).view(
                b, self.n_agents, self.n_actions
            )
            target = target.gather(2, actions).squeeze(-1)
            td_target = tensors["reward"] + self.gamma * (1 - tensors["done"]) * self.target_mixer(
                target, tensors["next_state"]
            )
        loss = F.smooth_l1_loss(total, td_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
        post = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(p.grad.detach()) for p in params if p.grad is not None]
            )
        )
        metrics = dict(
            loss=float(loss.detach()),
            td_abs_mean=float((td_target - total).detach().abs().mean()),
            q_total_mean=float(total.detach().mean()),
            q_total_min=float(total.detach().min()),
            q_total_max=float(total.detach().max()),
            target_q_mean=float(td_target.mean()),
            target_q_min=float(td_target.min()),
            target_q_max=float(td_target.max()),
            td_abs_p90=float(torch.quantile((td_target - total).detach().abs(), 0.9)),
            grad_norm_before_clip=float(norm),
            grad_norm_after_clip=float(post),
            lr=float(self.optimizer.param_groups[0]["lr"]),
        )
        metrics["nonfinite"] = not all(
            torch.isfinite(torch.tensor(v)).item() for v in metrics.values()
        )
        if metrics["nonfinite"]:
            raise FloatingPointError(f"Nonfinite learner update: {metrics}")
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in params):
            raise FloatingPointError("Nonfinite parameters after optimizer step")
        return metrics
