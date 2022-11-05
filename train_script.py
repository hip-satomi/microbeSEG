import argparse
import getpass
import json
import torch

from omero.gateway import BlitzGateway
from pathlib import Path
from shutil import rmtree

from src.training.train import CreateLabelsWorker, TrainWorker
from src.utils.data_export import DataExportWorker


def main():

    parser = argparse.ArgumentParser(description='microbeSEG training script')
    parser.add_argument('--omero_id', '-id', required=True, type=int, help='Training dataset id')
    parser.add_argument('--batch_size', '-b', default=4, type=int, help='Batch size')
    parser.add_argument('--iterations', '-i', default=1, type=int, help='Number of models to train')
    parser.add_argument('--method', '-m', default="distance", type=str, help='"boundary" or "distance"')
    parser.add_argument('--optimizer', '-o', default="Ranger", type=str, help='"Adam" or "Ranger"')
    parser.add_argument('--model_path', '-r', default=None, type=str, help='Model path for saving')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device to use')
    parser.add_argument('--username', default=None, type=str, help='OMERO username')
    parser.add_argument('--password', default=None, type=str, help='Better use the password prompt')
    parser.add_argument('--host', default=None, type=str, help='OMERO host')
    parser.add_argument('--port', default=None, type=str, help='OMERO port')
    args = parser.parse_args()

    # Get username and password
    omero_username = input("OMERO username") if args.username is None else args.username
    omero_password = getpass.getpass(prompt="Password") if args.password is None else args.password

    # Get host and port for the OMERO login
    with open(Path(__file__).parent / 'settings.json') as f:
        settings = json.load(f)
    omero_host = settings['omero_host'] if args.host is None else args.host
    omero_port = settings['omero_port'] if args.port is None else args.port

    # Check connection
    conn = BlitzGateway(omero_username, omero_password, host=omero_host, port=omero_port, secure=True)
    try:
        conn_status = conn.connect()
    except:
        raise Exception('No OMERO connection possible. Check inputs or connection.')
    else:
        if not conn_status:
            raise Exception('No OMERO connection possible. Check inputs or connection.')
    conn.close()

    # Paths
    model_path = (Path(__file__).parent / 'models') if args.model_path is None else Path(args.model_path)
    train_path = Path(__file__).parent / 'training_dataset'

    # Set device for using CPU or GPU
    if 'cuda' in args.device and not torch.cuda.is_available():
        raise ValueError('No cuda capable gpu device detected, use device "cpu"')
    device = torch.device(args.device)
    if 'cuda' in str(device):
        torch.backends.cudnn.benchmark = True

    # Get group id
    conn = BlitzGateway(omero_username, omero_password, host=omero_host, port=omero_port, secure=True)
    conn_status = conn.connect()
    conn.SERVICE_OPTS.setOmeroGroup("-1")  # not clear in which group: "-1" is "all groups"
    dataset = conn.getObject("Dataset", oid=args.omero_id)
    if dataset is None:
        raise Exception(f"Training set with id {args.omero_id} not found!")
    try:
        group_id = conn.getObject("Image", next(dataset.listChildren()).getId()).getDetails().group.id.val
    except StopIteration:
        print('No files found in training set')
        conn.close()
        return
    trainset_name = dataset.getName()
    conn.close()

    # Check method
    if not args.method.lower() in ["distance", "boundary"]:
        raise Exception(f"Unknown method {args.method}")

    # Check optimizer:
    if not args.optimizer.lower() in ["ranger", "adam"]:
        raise Exception(f"Unknown optimizer {args.optimizer}")

    trainset_path = train_path / trainset_name
    if trainset_path.is_dir():
        rmtree(str(trainset_path))
    trainset_path.mkdir(exist_ok=True)
    (trainset_path / 'train').mkdir(exist_ok=True)
    (trainset_path / 'val').mkdir(exist_ok=True)
    (trainset_path / 'test').mkdir(exist_ok=True)
    (model_path / trainset_name).mkdir(exist_ok=True)

    # Export training set from OMERO
    export_worker = DataExportWorker()
    print("Downloading data (pre-labeled but not corrected data are skipped)")
    export_worker.export_data(args.omero_id,
                              train_path,
                              omero_username,
                              omero_password,
                              omero_host,
                              omero_port,
                              group_id)
    mask_ids_train = list((train_path / trainset_name / 'train').glob('mask*.tif'))
    mask_ids_val = list((train_path / trainset_name / 'val').glob('mask*.tif'))
    if len(mask_ids_val) < 2 or len(mask_ids_train) < 2:
        print("   The training and the validation set should each contain at least two annotated images! Stop")
        return

    # Create labels
    print(f"Create {args.method} labels")
    label_worker = CreateLabelsWorker()
    label_worker.create_labels(trainset_path, args.method.lower())

    # Start training
    print(f"Start training")
    train_worker = TrainWorker()
    train_worker.start_training(trainset_path,
                                model_path / trainset_name,
                                args.method.lower(),
                                args.iterations,
                                args.optimizer.lower(),
                                args.batch_size,
                                device,
                                1,  # always 1 gpu
                                True)

    print('--- Finished ---')


if __name__ == "__main__":

    main()
