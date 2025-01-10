from abc import abstractmethod
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from properties import jacobian


from peft import get_peft_model
from adapters import peft_factory

from util import jacobian_vector_product

from time import time


class AcquisitionModel:
    @abstractmethod
    def get_encoder_dim(self, **kwargs):
        pass

    @abstractmethod
    def get_encoder(self, **kwargs):
        pass

    @abstractmethod
    def predict_probs(self, **kwargs):
        pass


class Transformer(nn.Module, AcquisitionModel):
    def __init__(self, name, args, meta, device, clf_token, peft=None):
        super().__init__()

        self.name = name
        self.args = args
        self.device = device
        self.clf_token = clf_token
        self.peft = peft

        model_cls = MODEL_CLS[meta.task_type]
        name = TRANSFORMERS[name]

        if peft:
            peft_config = peft_factory(peft)
            self.base_model = model_cls.from_pretrained(
                name, num_labels=meta.num_targets
            )
            self.classifier = get_peft_model(self.base_model, peft_config)
            self.classifier.print_trainable_parameters()
        else:
            self.classifier = model_cls.from_pretrained(
                name,
                num_labels=meta.num_targets,
            )
        self.num_targets = meta.num_targets
        self.num_hidden_layers = self.classifier.config.num_hidden_layers

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        output = self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states
        e = hidden_states[0].mean(dim=1)
        hidden = hidden_states[-1][:, self.clf_token, :]
        return_dict = {"embeddings": e, "encoded": hidden}

        return output, return_dict

    def encoded_layerwise_gradients(self, iter_):
        self.eval()
        grads_encoded = {i: [] for i in range(1, self.num_hidden_layers)}
        for batch_num, batch in enumerate(iter_):
            output = self.classifier(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_hidden_states=True,
            )
            layers = output.hidden_states[1:]
            for layer_num, layer in enumerate(layers[1:], 1):
                grad_enc = torch.autograd.grad(
                    layer[:, self.clf_token, :].sum(), layers[0], retain_graph=True
                )[0].sum((0, 1))
                grads_encoded[layer_num].append(grad_enc)

        grad_enc_tensors = {k: torch.stack(v).mean(0) for k, v in grads_encoded.items()}
        return grad_enc_tensors

    def encoded_layerwise_jacobians(self, iter_):
        self.eval()
        num_layers = self.classifier.config.num_hidden_layers
        grads_encoded = {i: [] for i in range(1, num_layers)}
        for batch_num, batch in enumerate(iter_):
            output = self.classifier(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_hidden_states=True,
            )
            layers = output.hidden_states[1:]
            for layer_num, layer in enumerate(layers[1:], 1):
                grad_enc = [
                    torch.autograd.grad(
                        layer[:, self.clf_token, j].sum(), layers[0], retain_graph=True
                    )[0].sum((0, 1))
                    for j in range(layer.shape[-1])
                ]
                grads_encoded[layer_num].append(torch.stack(grad_enc))
        grad_enc_tensors = {
            k: torch.stack(v).squeeze() for k, v in grads_encoded.items()
        }
        return grad_enc_tensors

    def embedding_layerwise_gradients(self, iter_):
        self.eval()
        num_layers = self.classifier.config.num_hidden_layers
        embeddings = self.classifier.get_input_embeddings()
        grads_embedding = {i: [] for i in range(self.num_hidden_layers)}

        for batch_num, batch in enumerate(iter_):
            x = batch.input_ids
            inputs_embeds = torch.autograd.Variable(embeddings(x), requires_grad=True)
            output = self.classifier(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                output_hidden_states=True,
            )
            layers = output.hidden_states[1:]
            for layer_num, layer in enumerate(layers):
                grad_emb = torch.autograd.grad(
                    layer.sum(), inputs_embeds, retain_graph=True
                )[0].sum((0, 1))
                grads_embedding[layer_num].append(grad_emb)

        grad_emb_tensors = {
            k: torch.stack(v).mean(0) for k, v in grads_embedding.items()
        }
        return grad_emb_tensors

    def layerwise_jacobian_norms(self, iter_):
        self.eval()
        embeddings = self.classifier.get_input_embeddings()
        num_layers = self.classifier.config.num_hidden_layers
        grads_encoded = {i: [] for i in range(num_layers)}

        for batch_num, batch in enumerate(iter_, 1):
            t = time()

            x = batch.input_ids
            inputs_embeds = torch.autograd.Variable(embeddings(x), requires_grad=True)
            output = self.classifier(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                output_hidden_states=True,
            )
            layers = output.hidden_states[1:]
            for layer_num, layer in enumerate(layers):
                layer_slice = layer[:, self.clf_token, :]
                jac_norm = self._estimate_jacobian_norm(layer_slice, inputs_embeds)
                jac_norm = jac_norm.cpu().item()
                grads_encoded[layer_num].append(jac_norm)
                torch.cuda.empty_cache()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, len(iter_), time() - t
                ),
                end="\r",
                flush=True,
            )

        grad_enc_tensors = {k: torch.tensor(v).mean() for k, v in grads_encoded.items()}
        return grad_enc_tensors

    def jachess_regularization(self, iter_, optimizer, reg_lambdas):
        """
        Implements JacHess regularization during the training of a neural network,
        specifically designed for transformer-like architectures. This regularization
        technique combines Jacobian and Hessian norms to smooth the model, considering
        both input-output relationships and second-order effects.

        Parameters:
        - iter_ (iterator): An iterator (like DataLoader) providing batches of data. Each
        batch should have `input_ids` for input IDs, and `attention_mask` for attention masks.
        - optimizer (Optimizer): A PyTorch optimizer object for model training.
        - reg_lambdas (list or array): Regularization lambda values, one for each layer in the model.

        The method sets the model to training mode, iterates through each batch, and computes
        JacHess regularization loss for each layer of the model. It updates the model parameters
        using the provided optimizer.

        Note:
        - Assumes `self.classifier` is a classifier model with `get_input_embeddings()` method.
        - Requires `self._estimate_jacobian_norm` and `self._estimate_hessian_norm` methods for
        estimating Jacobian and Hessian norms, respectively.
        - Designed for CUDA-enabled models (uses `torch.cuda.empty_cache()`).

        Use Case:
        Useful for enhancing the generalization capabilities of transformer-based models, especially
        in low-resource settings or when fine-tuning on small datasets.
        """

        self.train()
        embeddings = self.classifier.get_input_embeddings()

        # TODO
        jac_norm_list = []
        hess_norm_list = []

        print("JacHess regularization...")

        for batch_num, batch in enumerate(iter_, 1):
            t = time()
            optimizer.zero_grad()

            x = batch.input_ids
            inputs_embeds = torch.autograd.Variable(embeddings(x), requires_grad=True)
            output = self.classifier(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                output_hidden_states=True,
            )
            layers = output.hidden_states[1:]

            jachess_loss = []
            for layer_num in range(self.num_hidden_layers):
                inputs_embeds = torch.autograd.Variable(
                    embeddings(x), requires_grad=True
                )
                output = self.classifier(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch.attention_mask,
                    output_hidden_states=True,
                )
                layers = output.hidden_states[1:]
                layer = layers[layer_num]

                layer_slice = layer[:, self.clf_token, :]
                jac_norm = self._estimate_jacobian_norm(layer_slice, inputs_embeds).to(
                    self.device
                )
                hess_norm = self._estimate_hessian_norm(layer_slice, inputs_embeds).to(
                    self.device
                )
                jachess_loss = reg_lambdas[layer_num] * (jac_norm + hess_norm.mean())
                jachess_loss.backward()

            optimizer.step()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, len(iter_), time() - t
                ),
                end="\r",
                flush=True,
            )

        torch.cuda.empty_cache()

    def _estimate_jacobian_norm(self, layer, inputs_embeds, n_proj=10, top_k=1):
        """
        Estimate the squared Frobenius norm of the Jacobian for a given layer
        with respect to input embeddings.

        This method computes an estimate of the squared Frobenius norm of the
        Jacobian matrix for a neural network layer. The estimation is done using
        random projections v ~ N(0,1) to calculate the Jacobian-vector products (JVPs).
        The norms of these JVPs are used to estimate the squared Frobenius norm of the Jacobian.
        The method returns the mean of these estimated norms along with the indices of the top_k
        highest values in this mean.

        Parameters:
        layer (torch.nn.Module): The neural network layer for which the Jacobian is
                                 to be estimated.
        inputs_embeds (torch.Tensor): The input embeddings to the layer. These are
                                    the points at which the Jacobian is estimated.
        n_proj (int, optional): The number of random projections used for estimating
                                the Jacobian.
        top_k (int, optional): The number of top indices to select from the mean of
                               the estimated norms.

        Returns:
        tuple: A tuple containing two elements:
            - estimated_norm (torch.Tensor): The estimated mean squared Frobenius norm
                of the Jacobian.
            - top_k_norms (torch.Tensor): The indices of the top_k highest values in
                the mean of the estimated norms.

        """
        norm_proj = []
        for i in range(n_proj):
            v = torch.randn(layer.shape).to(self.device)
            jvp = jacobian_vector_product(layer, inputs_embeds, v, create_graph=True)
            norm_proj.append(jvp.cpu())

        norm_tensor = torch.stack(norm_proj)
        jacobian_norms = norm_tensor.norm(p="fro", dim=-1).square()
        estimated_norm = jacobian_norms.mean()
        # top_k_norms = torch.topk(jacobian_norms.mean(dim=0), k=top_k).indices

        return estimated_norm

    def _estimate_hessian_norm(self, layer, inputs_embeds, k=10, n_proj=1):
        k_norms = []

        indices = torch.randint(
            size=[k], low=0, high=layer.shape[1], device=self.device
        )
        for index in indices:
            norm_proj = []
            layer_slice = torch.gather(
                layer, 1, index.repeat(layer.shape[0]).unsqueeze(1)
            )
            grads = torch.autograd.grad(
                layer_slice.sum(), inputs_embeds, create_graph=True
            )[0]
            for i in range(n_proj):
                v = torch.randn(grads.shape).to(self.device)
                hvp = jacobian_vector_product(grads, inputs_embeds, v)
                norm_proj.append(hvp.cpu())
            norm_tensor = torch.stack(norm_proj)
            hessian_norms = norm_tensor.norm(p="fro", dim=-1).square()
            estimated_norm = hessian_norms.mean()
            k_norms.append(estimated_norm)

        k_norms_tensor = torch.tensor(k_norms)
        return k_norms_tensor

    def layerwise_regularization(
        self,
        output,
        base_output,
    ):
        hidden = output.hidden_states[1:]
        base_hidden = base_output.hidden_states[1:]

        reg_sum = []

        for new_rep, base_rep in zip(hidden, base_hidden):
            new_cls = new_rep[:, self.clf_token, :].squeeze()
            base_cls = base_rep[:, self.clf_token, :].squeeze()
            rep_diff = new_cls - base_cls
            diff_norm = rep_diff.norm(p="fro", dim=1)

            reg_sum.append(diff_norm.mean())

        reg_sum_t = sum(reg_sum)
        mean_norm = reg_sum_t.mean()
        return mean_norm

    def predict_probs(self, inputs, attention_mask=None, token_type_ids=None):
        with torch.inference_mode():
            output, _ = self(
                inputs, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            logits = output.logits
            if self.num_targets == 1:
                # Binary classification
                y_pred = torch.sigmoid(logits)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
            else:
                # Multiclass classification
                y_pred = F.softmax(logits, dim=1)
            return y_pred

    def get_encoder_dim(self):
        return self.classifier.config.hidden_size

    def get_encoded(self, inputs, attention_mask=None, token_type_ids=None):
        with torch.inference_mode():
            output = self.classifier(
                inputs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )
            hidden = output.hidden_states[-1][:, self.clf_token, :]
            return hidden

    def get_classifier_name(self):
        return TRANSFORMER_CLASSIFIERS[self.name]


def initialize_model(args, meta, device):
    model = Transformer(
        name=args.model,
        args=args,
        meta=meta,
        device=device,
        clf_token=CLF_TOKEN[args.model],
        peft=args.peft,
    )

    return model


TRANSFORMERS = {
    "BERT": "bert-base-uncased",
    "ELECTRA": "google/electra-base-discriminator",
    "OPT-125m": "facebook/opt-125m",
    "OPT-350m": "facebook/opt-350m",
    "OPT-1.3b": "facebook/opt-1.3b",
    "OPT-2.7b": "facebook/opt-2.7b",
    "OPT-6.7b": "facebook/opt-6.7b",
    "GPTJ": "EleutherAI/gpt-j-6b",
    "LLaMA-2-7b": "llama-2-7b",
}

CLF_TOKEN = {
    "BERT": 0,
    "ELECTRA": 0,
    "OPT-125m": -1,
    "OPT-350m": -1,
    "OPT-1.3b": -1,
    "OPT-2.7b": -1,
    "OPT-6.7b": -1,
    "GPTJ": -1,
    "LLaMA-2-7b": -1,
}

TRANSFORMER_CLASSIFIERS = {
    "BERT": "bert",
    "ALBERT": "albert",
    "ELECTRA": "electra",
    "OPT-125m": "opt",
    "OPT-350m": "opt",
    "OPT-1.3b": "opt",
    "OPT-2.7b": "opt",
    "OPT-6.7b": "opt",
    "LLaMA-2-7b": "llama",
}


MODEL_CLS = {
    "clf": AutoModelForSequenceClassification,
    "reg": AutoModelForSequenceClassification,
    "seq": AutoModelForTokenClassification,
}
