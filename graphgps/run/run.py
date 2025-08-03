from graphgps.run.args import TrainArgs, PredictArgs
from graphgps.run.utils import *
from graphgps.data.data import DatasetFromCSVFile
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd


def graphgps_train(arguments=None):
    from torch_geometric.graphgym.config import cfg
    args = TrainArgs().parse_args(arguments)
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    # modify cfg based on the arguments
    cfg.out_dir = args.save_dir
    cfg.dataset.data_path = args.data_path
    cfg.dataset.smiles_columns = args.smiles_columns
    cfg.dataset.target_columns = args.target_columns
    cfg.dataset.features_columns = args.features_columns
    cfg.dataset.features_generator = args.features_generator
    cfg.dataset.separate_test_path = args.separate_test_path
    cfg.dataset.separate_val_path = args.separate_val_path
    cfg.output_details = args.output_details
    if args.features_generator is not None or cfg.dataset.features_columns is not None:
        cfg.gnn.use_features = True
        n_features = 0
        if args.features_generator is not None:
            for fg in args.features_generator:
                if fg in ['rdkit_2d', 'rdkit_2d_normalized']:
                    n_features += 200
                elif fg in ['morgan', 'morgan_count']:
                    n_features += 2048
                else:
                    raise ValueError(f"Unknown features generator: {fg}")
            n_features *= len(args.smiles_columns)
        if cfg.dataset.features_columns is not None:
            n_features += len(cfg.dataset.features_columns)
        cfg.gnn.n_features = n_features
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(args.n_jobs)
    for run_id, split_index in enumerate(args.split_index_list):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        if os.path.exists(f'{cfg.run_dir}/ckpt/{cfg.optim.max_epoch - 1}.ckpt'):
            continue
        set_printing()
        cfg.dataset.split_idxs = split_index
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}.")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)


def graphgps_predict(arguments=None):
    from torch_geometric.graphgym.config import cfg
    args = PredictArgs().parse_args(arguments)
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    if args.features_generator is not None or args.features_columns is not None:
        cfg.gnn.use_features = True
        n_features = 0
        if args.features_generator is not None:
            for fg in args.features_generator:
                if fg in ['rdkit_2d', 'rdkit_2d_normalized']:
                    n_features += 200
                elif fg in ['morgan', 'morgan_count']:
                    n_features += 2048
                else:
                    raise ValueError(f"Unknown features generator: {fg}")
            n_features *= len(args.smiles_columns)
        if cfg.dataset.features_columns is not None:
            n_features += len(cfg.dataset.features_columns)
        cfg.gnn.n_features = n_features

    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(args.n_jobs)
    # Set Model
    auto_select_device()
    model = create_model()
    model = init_model_from_pretrained(
        model, args.checkpoint_dir, freeze_main=True,
        reset_prediction_head=False
    )
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    # Prepare data
    dataset = DatasetFromCSVFile(data_path=args.test_path,
                                 smiles_columns=args.smiles_columns,
                                 features_generator=args.features_generator,
                                 root=cfg.out_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size,shuffle=False)
    # Make predictions
    predictions = []
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            batch.to(torch.device(cfg.accelerator))
            output = model(batch)
            predictions.append(output[0].detach().to('cpu', non_blocking=True))
    predictions = np.concatenate(predictions)
    df = pd.read_csv(args.test_path)
    df['prediction'] = predictions
    df.to_csv(args.preds_path, index=False)
