from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

logging_dir = None

writer = None

def create_writer():

    global logging_dir

    datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

    if logging_dir is None:
        runs_dir = Path("./") / Path(f"runs/")
        runs_dir.mkdir(exist_ok = True)

        logging_dir = runs_dir / Path(f"{datetime_str}")

        logging_dir.mkdir(exist_ok = True)
    logging_dir = str(logging_dir.absolute())

    global writer

    writer = SummaryWriter(log_dir=logging_dir)

def log_scale_train(epoch, loss, acc):
    global writer
    if writer is None:
        create_writer()
    writer.add_scalar('Train Loss', loss, epoch)
    writer.add_scalar('Valid Accuracy', acc, epoch)

def log_scale_test(epoch, loss, acc):
    global writer
    if writer is None:
        create_writer()

    writer.add_scalar('Loss/test', loss, epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)

def log_learning_rate(optimizer, epoch):
    global writer
    if writer is None:
        create_writer()
    for param_group in optimizer.param_groups:
        writer.add_scalar("Learning Rate", param_group['lr'], epoch)

        