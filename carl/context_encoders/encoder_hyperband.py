import torch as th
import numpy as np
import os
from tqdm import tqdm
import pickle
from mle_hyperopt import HyperbandSearch

from carl.context_encoders import ContextEncoder, ContextAE, ContextVAE, ContextBVAE, ContextAESigmoid
import json
import shutil

import hydra
from omegaconf import DictConfig

import pandas as pd
from sklearn.model_selection import KFold

step = 0
base_dir = os.getcwd()


@hydra.main("./configs", "hyperband")
def main(cfg: DictConfig) -> None:

    global base_dir

    out_dir = os.path.join(base_dir, cfg.encoder.outdir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not cfg.encoder.context_dataset:
        raise ValueError(
            "Please specify a context dataset. To generate a context dataset, use the 'generate_context_dataset.py' after generating a contexts_train.json generated by running the carl_env with given specs."
        )

    dataset = np.load(os.path.join(base_dir, cfg.encoder.context_dataset))
    kfold = KFold(n_splits=cfg.encoder.kfolds, shuffle=True)

    training_set = []
    validation_set = []
    for train_index, val_index in kfold.split(dataset):
        training_set.append(dataset[train_index])
        validation_set.append(dataset[val_index])

    # Blackbox objective
    def train_model(**kwargs):

        global step

        print('\n')
        print(f"Iter: {step}")
        for key in kwargs:
            print(key, kwargs[key])

        iter_dir = os.path.join(out_dir, f"iter_{step}")
        if not os.path.exists(iter_dir):
            os.mkdir(os.path.join(iter_dir))
        else:
            shutil.rmtree(os.path.join(iter_dir))
            os.mkdir(os.path.join(iter_dir))

        

        final_train_losses = []
        mean_val_losses = []
        models = []

        for idx in tqdm(range(cfg.encoder.kfolds)):
            
            # Generate the train and val sets from the fold ids
            train_set = training_set[idx]
            val_set = validation_set[idx]

            # Create a new model
            encoder_cls = eval(cfg.encoder.model)
            model = encoder_cls(
                cfg.encoder.input_dim, 
                cfg.encoder.latent_dim, 
                cfg.encoder.hidden_dim
            )
            
            # Create a new optimizer
            optimizer = th.optim.Adam(
                model.parameters(), 
                lr=kwargs["rate"], 
                weight_decay=1e-8
            )


            loader = th.utils.data.DataLoader(
                dataset=train_set, batch_size=kwargs["batch"], shuffle=False
            )

            losses = []
            for _ in range(cfg.encoder.epochs):

                for vector in loader:

                    # Output of Autoencoder
                    results = model(vector)

                    # Get the loss
                    loss_dict = model.loss_function(*results, M_N=cfg.encoder.M_N)
                    loss = loss_dict["loss"]

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Storing the losses in a list for plotting
                    losses.append(loss.item())

            final_train_losses.append(losses[-1])

            ## Validation
            val_loader = th.utils.data.DataLoader(
                dataset=val_set, batch_size=kwargs["batch"], shuffle=False
            )

            val_losses = []
            for _ in range(cfg.encoder.val_epochs):
                for vector in val_loader:

                    # Output of Autoencoder
                    val_results = model(vector)

                    # Get hte validation loss
                    val_loss_dict = model.loss_function(*val_results, M_N=cfg.encoder.M_N)
                    val_loss = val_loss_dict["loss"]

                    # Storing the losses in a list for plotting
                    val_losses.append(val_loss.item())

            mean_val_losses.append(np.mean(val_losses))

            models.append(model)

        print(f"Final Train Losses: {final_train_losses}")
        print(f"Mean Validation Losses: {(mean_val_losses)}")

        # Save the model with the best validation loss
        print(f'Saving the model with the best validation loss at index: {np.nanargmin(mean_val_losses)}')
        th.save(models[np.nanargmin(mean_val_losses)], os.path.join(iter_dir, "model.zip"))

        with open(os.path.join(iter_dir, "losses.pkl"), "wb") as f:
            pickle.dump(losses, f)

        step = step + 1

    # Create a sampling strategy
    strategy = HyperbandSearch(
        real={
            "rate": {
                "begin": cfg.hyperband.real.rate.begin,
                "end": cfg.hyperband.real.rate.end,
                "prior": cfg.hyperband.real.rate.prior,
            },
            # "decay": {
            #     "begin": cfg.hyperband.real.decay.begin,
            #     "end": cfg.hyperband.real.decay.end,
            #     "prior": cfg.hyperband.real.decay.prior,
            # },
        },
        integer={
            "batch": {
                "begin": cfg.hyperband.integer.batch.begin,  #
                "end": cfg.hyperband.integer.batch.end,
                "prior": cfg.hyperband.integer.batch.prior,
            }
        },
        search_config={
            "max_resource": cfg.hyperband.search_config.max_resource,
            "eta": cfg.hyperband.search_config.eta,
        },
        seed_id=cfg.hyperband.seed,
    )

    # Generate the configs using the strategy and dump them
    configs = strategy.ask()
    with open(os.path.join(out_dir, "configs.json"), "w") as f:
        json.dump(configs, f, indent=4)


    # Train the model with the generated configs
    for c in configs:
        train_model(**c["params"])

   
if __name__ == "__main__":
    main()
