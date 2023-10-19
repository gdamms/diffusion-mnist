import torch
import torch.utils.data

import rich.progress

from typing import *


class TrainProgress(rich.progress.Progress):
    """A progress bar which tracks the progress of training epochs."""

    def __init__(
        self: 'TrainProgress',
        nb_epochs: int,
        epoch_size: int,
        *columns: str | rich.progress.ProgressColumn,
        console: rich.progress.Console | None = None,
        auto_refresh: bool = True,
        refresh_per_second: float = 10,
        speed_estimate_period: float = 30,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        get_time: rich.progress.GetTimeCallable | None = None,
        disable: bool = False,
        expand: bool = False,
    ) -> None:
        """Initialize the progress bar.

        Args:
            nb_epochs (int): The number of epochs.
            epoch_size (int): The size of each epoch.
            *columns (str | rich.progress.ProgressColumn): The columns to display.
            console (rich.progress.Console, optional): The console to use. Defaults to None.
            auto_refresh (bool, optional): Whether to automatically refresh the progress bar. Defaults to True.
            refresh_per_second (float, optional): The number of times to refresh the progress bar per second. Defaults to 10.
            speed_estimate_period (float, optional): The number of seconds to use when estimating the speed. Defaults to 30.
            transient (bool, optional): Whether to use transient mode. Defaults to False.
            redirect_stdout (bool, optional): Whether to redirect stdout. Defaults to True.
            redirect_stderr (bool, optional): Whether to redirect stderr. Defaults to True.
            get_time (rich.progress.GetTimeCallable, optional): A callable which returns the current time. Defaults to None.
            disable (bool, optional): Whether to disable the progress bar. Defaults to False.
            expand (bool, optional): Whether to expand the progress bar. Defaults to False.
        """
        self.nb_epochs = nb_epochs
        self.epoch_size = epoch_size
        super().__init__(
            *columns,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            speed_estimate_period=speed_estimate_period,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_time=get_time,
            disable=disable,
            expand=expand,
        )
        self.epoch_tasks = []
        self.total_task = self.add_task("total", progress_type="total", total=nb_epochs*epoch_size)
        self.values = {}

    def get_renderables(self: 'TrainProgress'):
        """Override the default renderables to display the epoch number."""
        pad = len(f"{self.nb_epochs}")
        for task in self.tasks:
            # The total task.
            if task.fields.get("progress_type") == "total":
                self.columns = (
                    f"Training:",
                    rich.progress.BarColumn(),
                    f"{len(self.epoch_tasks):{pad}}/{self.nb_epochs}",
                    "•",
                    rich.progress.TimeRemainingColumn(),
                )

            # The epoch tasks.
            if task.fields.get("progress_type") == "epoch":
                epoch_id = task.fields.get("epoch_id")
                self.columns = (
                    f"Epoch {epoch_id:{pad}}:",
                    rich.progress.BarColumn(),
                    f"{task.completed}/{task.total}",
                    "•",
                    rich.progress.TimeElapsedColumn(),
                    '•',
                    ' | '.join(f"{key}: {value[-1]:.4f} " for key, value in self.values.items()),
                )

            yield self.make_tasks_table([task])

    def new_epoch(self: 'TrainProgress'):
        """Create a new epoch task."""
        epoch_task = self.add_task(
            "epoch",
            progress_type="epoch",
            epoch_id=len(self.epoch_tasks) + 1,
            total=self.epoch_size,
        )
        self.epoch_tasks.append(epoch_task)

    def step(self: 'TrainProgress', count: int = 1):
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.
        """
        if len(self.epoch_tasks) == 0 or self.tasks[self.epoch_tasks[-1]].completed == self.epoch_size:
            self.new_epoch()
        self.update(self.epoch_tasks[-1], advance=count)
        self.update(self.total_task, advance=count)

    def new_values(self: 'TrainProgress', **values: Any):
        """Update the progress bar with new values.

        Args:
            **values (Any): The values to update the progress bar with.
        """
        for key, value in values.items():
            current_value = self.values.get(key, [])
            self.values[key] = current_value + [value]


class Trainer:
    """A class which trains models."""

    def __init__(self):
        """Initialize the trainer."""
        self.progress: TrainProgress | None = None

    def train(
        self: 'Trainer',
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        val_loader: torch.utils.data.DataLoader | None = None,
    ):
        """Train the model for the given number of epochs.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): The training dataset.
            epochs (int): The number of epochs to train the model for.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            device (torch.device, optional): The device to use. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
            val_loader (torch.utils.data.DataLoader, optional): The validation dataset. Defaults to None.
        """
        with TrainProgress(
            nb_epochs=epochs,
            epoch_size=len(train_loader),
        ) as progress:
            self.progress = progress

            for _ in range(epochs):
                self.train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                )
                if val_loader:
                    self.validate(
                        model,
                        val_loader,
                        criterion,
                        device,
                    )

    def train_epoch(
        self: 'Trainer',
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
    ):
        """Train the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): The training dataset.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            device (torch.device): The device to use.
        """
        model.train()
        for batch in train_loader:
            # Move the batch to the device.
            batch = [b.to(device) for b in batch]

            # Seprarate the inputs and labels.
            inputs = batch[:-1]
            labels = batch[-1]

            # Train the model.
            optimizer.zero_grad()
            output = model(*inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Update the progress bar.
            self.progress.new_values(loss=loss.item())
            self.progress.step()

    def validate(
        self: 'Trainer',
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
    ):
        """Validate the model on the given validation dataset.

        Args:
            model (torch.nn.Module): The model to validate.
            val_loader (torch.utils.data.DataLoader): The validation dataset.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            device (torch.device): The device to use.
        """
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                output = model(batch[:-1])
                loss = criterion(output, batch[-1])
