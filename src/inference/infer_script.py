import argparse
import getpass
import json
import torch

from copy import copy
from omero.gateway import BlitzGateway
from pathlib import Path

from src.inference.infer import InferWorker


def main():

    parser = argparse.ArgumentParser(description='microbeSEG inference script')
    parser.add_argument('--omero_ids', '-ids', required=True, type=int, nargs='+',
                        help='Project ids or dataset ids or file ids (need to be from same group)')
    parser.add_argument('--id_type', '-i', required=True, type=str, help='"project" or "dataset" or "file"')
    parser.add_argument('--model', '-m', required=True, type=str, help='Path to model')
    parser.add_argument('--thresholds', '-t', default=[0.10, 0.45], type=float, nargs='+', help='Thresholds for distance models')
    parser.add_argument('--result_path', '-r', default=None, type=str, help='Result path')
    parser.add_argument('--channel', '-c', default=0, type=int, help='Channel to process')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device to use')
    parser.add_argument('--overwrite', '-o', default=False, action='store_true', help='Overwrite existing results')
    parser.add_argument('--upload', '-u', default=False, action='store_true', help='Upload results to OMERO')
    parser.add_argument('--username', default=None, type=str, help='OMERO username')
    parser.add_argument('--password', default=None, type=str, help='Better use the password prompt')
    parser.add_argument('--host', default=None, type=str, help='OMERO host')
    parser.add_argument('--port', default=None, type=str, help='OMERO port')
    args = parser.parse_args()

    # Get username and password
    omero_username = input("OMERO username") if args.username is None else args.username
    omero_password = getpass.getpass(prompt="Password") if args.password is None else args.password

    # Get host and port for the OMERO login
    with open(Path(__file__).parents[2] / 'settings.json') as f:
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

    # Result path
    result_path = (Path(__file__).parents[2] / 'results') if args.result_path is None else Path(args.result_path)

    # Check if model is available
    inference_model = Path(args.model)
    if not (inference_model.parent / f"{inference_model.stem}.pth").is_file():
        raise Exception(f'{inference_model.parent / f"{inference_model.stem}.pth"} not found!')
    if not (inference_model.parent / f"{inference_model.stem}.json").is_file():
        raise Exception(f'{inference_model.parent / f"{inference_model.stem}.json"} not found!')
    inference_model = inference_model.parent / f"{inference_model.stem}.json"

    # Check thresholds
    if len(args.thresholds) != 2:
        raise Exception(f"{len(args.thresholds)} threshold given, needed are 2")

    # Set device for using CPU or GPU
    if 'cuda' in args.device and not torch.cuda.is_available():
        raise ValueError('No cuda capable gpu device detected, use device "cpu"')
    device = torch.device(args.device)
    if 'cuda' in str(device):
        torch.backends.cudnn.benchmark = True

    # Get all file ids for selected project or dataset
    conn = BlitzGateway(omero_username, omero_password, host=omero_host, port=omero_port, secure=True)
    conn_status = conn.connect()
    conn.SERVICE_OPTS.setOmeroGroup("-1")  # not clear in which group: "-1" is "all groups"
    group_ids, file_ids = [], []
    if args.id_type.lower() == 'project':
        for project_id in args.omero_ids:
            project = conn.getObject("Project", oid=project_id)
            for dataset in project.listChildren():
                for file in dataset.listChildren():
                    file_ids.append(file.getId())
                    group_ids.append(conn.getObject("Image", file.getId()).getDetails().group.id.val)
    elif args.id_type.lower() == 'dataset':
        for dataset_id in args.omero_ids:
            dataset = conn.getObject("Dataset", oid=dataset_id)
            for file in dataset.listChildren():
                file_ids.append(file.getId())
                group_ids.append(conn.getObject("Image", file.getId()).getDetails().group.id.val)
    elif args.id_type.lower() == 'file':
        for file_id in args.omero_ids:
            file = conn.getObject("Image", oid=file_id)
            file_ids.append(file.getId())
            group_ids.append(file.getDetails().group.id.val)
    else:
        raise Exception(f'Unknown ID type {args.id_type}')
    conn.close()

    if len(file_ids) == 0:
        print('No files found')
        return

    group_ids = set(group_ids)
    if len(group_ids) > 1:
        raise Exception("Select only projects, datasets, and files from the same group!")

    # Start inference
    infer_worker = InferWorker(copy(file_ids),
                               result_path,
                               omero_username,
                               omero_password,
                               omero_host,
                               omero_port,
                               group_ids.pop(),
                               inference_model,
                               device,
                               args.thresholds,
                               args.channel,
                               args.upload,
                               args.overwrite,
                               False,
                               True)

    infer_worker.start_inference()

    print('--- Finished ---')


if __name__ == "__main__":

    main()
