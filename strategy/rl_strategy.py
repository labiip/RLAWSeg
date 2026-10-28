

import torch
import numpy as np
from collections import deque
from .rl_agent import DQNAgent
from .rl_utils import load_single_image, compute_boundary_uncertainty, compute_diversity
import random
from strategy.ap_strategy import compute_score
class RLActiveLearningSampler:


    def __init__(self, state_dim=10, hidden_dim=128, lr=1e-4, device='cuda'):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device


        self.agent = None
        self.env = None
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.representations = []  

        
        self.scale = None
        self.input_size = (224, 224)
        self.num_classes = 4

        
        self.history = {
            'performance': [],
            'labeled_ratio': [],
            'rewards': [],
            'selected_samples': []
        }
        self.current_round = 0
        self.max_rounds = 10

    def initialize(self, seg_net, segnet1, predictor, unlabeled_pool, val_loader,
                   budget_per_round, max_rounds, sample_weight=None, score_type='mean',
                   scale=None, input_size=(224, 224), num_classes=4):

        self.seg_net = seg_net
        self.segnet1 = segnet1
        self.predictor = predictor
        self.unlabeled_pool = unlabeled_pool.copy()
        self.labeled_pool = []
        self.val_loader = val_loader
        self.budget_per_round = budget_per_round
        self.max_rounds = max_rounds
        self.sample_weight = sample_weight
        self.score_type = score_type


        self.scale = scale
        self.input_size = input_size
        self.num_classes = num_classes


        self._register_hook()


        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=len(unlabeled_pool),
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            device=self.device
        )

        
        self.current_round = 0
        self.history = {'performance': [], 'labeled_ratio': [], 'rewards': [], 'selected_samples': []}

        print(f"[RL] : c={len(unlabeled_pool)}, y={budget_per_round}")

    def _register_hook(self):


        def hook_fn(module, input, output):

            self.representations.append(
                self.avgpool(output[-1]).detach().cpu().numpy().squeeze(axis=(2, 3))
            )


        if hasattr(self.seg_net, 'encoder'):
            self.handle = self.seg_net.encoder.register_forward_hook(hook_fn)

    def _get_state(self):

        if len(self.unlabeled_pool) == 0:
            return np.zeros(self.state_dim)


        sample_size = min(30, len(self.unlabeled_pool))
        sampled_paths = random.sample(self.unlabeled_pool, sample_size)

        uncertainties = []
        boundary_scores = []
        self.representations = []  

        self.seg_net.eval()
        with torch.no_grad():
            for path in sampled_paths:

                image_np = load_single_image(path, self.scale, self.input_size)
                image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(self.device)


                output, _ = self.seg_net(image_tensor)


                from strategy.ap_strategy import boundary_perturbation_score_v2
                boundary_score = boundary_perturbation_score_v2(output[0])
                boundary_scores.append(boundary_score)


                if self.predictor is not None:
                    try:
                        predictor_data = torch.stack([
                            torch.cat((image_tensor[i], torch.softmax(output[i], dim=0)), dim=0)
                            for i in range(image_tensor.shape[0])
                        ], dim=0)
                        pred_out = self.predictor(predictor_data)
                        from strategy.ap_strategy import compute_score
                        scores = compute_score(pred_out, self.sample_weight, self.score_type)

                        uncertainties.extend(scores)
                    except Exception as e:
                        print(f"[RL] predictor error: {e}")
                        pass

        #
        features = None
        if self.representations:
            features = np.concatenate(self.representations, axis=0)
            if len(features) > sample_size:
                features = features[:sample_size]

        state = [
            self.current_round / self.max_rounds,  
            len(self.labeled_pool) / (len(self.labeled_pool) + len(self.unlabeled_pool) + 1e-8),  
            np.mean(uncertainties) if uncertainties else 0.5,  
            np.std(uncertainties) if uncertainties else 0.0,  
            np.mean(boundary_scores) if boundary_scores else 0.5,  
            compute_diversity(features) if features is not None else 0.0,  
            self._compute_performance_trend(),  
            len(self.unlabeled_pool) / (len(self.labeled_pool) + len(self.unlabeled_pool) + 1e-8),  
        ]


        perf_history = self.history['performance'][-3:] if self.history['performance'] else [0.5, 0.5, 0.5]
        while len(perf_history) < 3:
            perf_history.append(0.5)
        state.extend(perf_history)


        if len(state) < self.state_dim:
            state.extend([0.0] * (self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]

        return np.array(state, dtype=np.float32)

    def _compute_performance_trend(self):

        if len(self.history['performance']) < 2:
            return 0.0
        y = np.array(self.history['performance'][-5:])
        if len(y) < 2 or np.std(y) < 1e-8:
            return 0.0
        x = np.arange(len(y))
        z = np.polyfit(x, y, 1)
        return float(z[0])

    def sample(self, seg_net, segnet1, predictor, unlabeled_data_pool,
               sample_loader, sample_nums, sample_count,
               sample_weight=None, al_mode='rl', score_type='mean'):

        k = int(sample_nums)

        if self.agent is None:
            print("[RL] 警告：采样器未初始化，使用随机采样")
            indices = np.random.choice(len(unlabeled_data_pool), min(k, len(unlabeled_data_pool)), replace=False)
            labeled_path = [unlabeled_data_pool[i] for i in indices]
            return labeled_path, []


        self.unlabeled_pool = unlabeled_data_pool

        
        state = self._get_state()
        self._last_state = state.copy()

        
        if sample_count < 3:  
            indices = np.random.choice(len(unlabeled_data_pool), min(k, len(unlabeled_data_pool)), replace=False)
            selected_indices = indices
            selected_q_values = None  
        else:

            selected_indices, selected_q_values = self.agent.select_topk_actions(
                state,
                k=min(k, len(unlabeled_data_pool)),
                valid_actions=len(unlabeled_data_pool)
            )


        if 'q_values' not in self.history:
            self.history['q_values'] = []


        if selected_q_values is not None:
            self.history['q_values'].append(selected_q_values)


        selected_indices = [min(i, len(unlabeled_data_pool) - 1) for i in selected_indices]
        selected_indices = list(dict.fromkeys(selected_indices))[:k]


        labeled_path = [unlabeled_data_pool[i] for i in selected_indices]


        self.history['selected_samples'].append(labeled_path)
        self._last_action = np.array(selected_indices[:1])[0] if selected_indices else 0
        self.current_round += 1

        return labeled_path, []

    def get_diversity_history(self):

        diversity_per_round = []

        if 'selected_samples' not in self.history or len(self.history['selected_samples']) == 0:
            return diversity_per_round

        for selected_paths in self.history['selected_samples']:
            if len(selected_paths) < 2:
                diversity_per_round.append(0.0)
                continue


            features = []
            self.seg_net.eval()
            with torch.no_grad():
                for path in selected_paths:
                    from .rl_utils import load_single_image
                    image_np = load_single_image(path, self.scale, self.input_size)
                    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(self.device)


                    self.representations = []
                    _ = self.seg_net(image_tensor)
                    if self.representations:

                        feat = self.representations[-1].flatten()
                        if hasattr(feat, 'cpu'):
                            feat = feat.cpu().numpy()
                        features.append(feat)

            if len(features) >= 2:

                features = np.array(features)
                diversity = 0.0
                n = len(features)
                for i in range(n):
                    for j in range(i + 1, n):
                        diversity += np.linalg.norm(features[i] - features[j])
                diversity = 2 * diversity / (n * (n - 1)) if n > 1 else 0.0
                diversity_per_round.append(diversity)
            else:
                diversity_per_round.append(0.0)

        return diversity_per_round

    def update_reward(self, new_performance):

        if self.agent is None:
            return 0.0


        old_performance = self.history['performance'][-1] if self.history['performance'] else 0.0
        reward = new_performance - old_performance


        self.history['performance'].append(new_performance)
        self.history['rewards'].append(reward)

        if hasattr(self, '_last_state') and hasattr(self, '_last_action'):
            self.agent.memory.push(
                self._last_state,
                self._last_action,
                reward,
                self._get_state(),  
                False  
            )

            
            loss = self.agent.learn()

            
            delattr(self, '_last_state')
            delattr(self, '_last_action')

            return loss

        return 0.0

    def save(self, path):
 
        if self.agent is not None:
            self.agent.save(path)

    def load(self, path):

        if self.agent is not None:
            self.agent.load(path)
