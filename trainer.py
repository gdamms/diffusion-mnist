import torch
import torch.utils.data

import rich.progress

from typing import *


class TrainProgress(rich.progress.Progress):
    """A progress bar which tracks the progress of training epochs."""

    def __init__(
        self: 'TrainProgress',
        nb_epochs: int,
        train_size: int,
        val_size: int = 0,
        test_size: int = 0,
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
            train_size (int): The size of each tain epoch.
            val_size (int, optional): The size of each validation epoch. Defaults to 0.
            test_size (int, optional): The size of the test epoch. Defaults to 0.
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
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
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
        self.train_tasks = []
        self.val_tasks = []
        self.test_task = None
        self.total_task = self.add_task(
            "total",
            progress_type="total",
            total=nb_epochs * (train_size + val_size) + test_size,
        )
        self.train_values = []
        self.val_values = []
        self.test_values = {}

    def get_renderables(self: 'TrainProgress'):
        """Override the default renderables to display the epoch number."""
        pad = len(f"{self.nb_epochs}")
        for task in self.tasks:
            # The total task.
            if task.fields.get("progress_type") == "total":
                self.columns = (
                    f"Working:",
                    rich.progress.BarColumn(),
                    f"{len(self.train_tasks):{pad}}/{self.nb_epochs}",
                    "•",
                    rich.progress.TimeRemainingColumn(),
                )

            # The train tasks.
            if task.fields.get("progress_type") == "train":
                epoch_id = task.fields.get("epoch_id")
                self.columns = (
                    f"Train {epoch_id:{pad}}:",
                    rich.progress.BarColumn(),
                    f"{task.completed}/{task.total}",
                    "•",
                    rich.progress.TimeElapsedColumn(),
                    '•',
                    ' | '.join(
                        f"{key}: {value[-1]:.4f}" for key, value in self.train_values[epoch_id-1].items()),
                )

            # The val tasks.
            if task.fields.get("progress_type") == "val":
                epoch_id = task.fields.get("epoch_id")
                self.columns = (
                    f"Val {epoch_id:{pad}}:",
                    rich.progress.BarColumn(),
                    f"{task.completed}/{task.total}",
                    "•",
                    rich.progress.TimeElapsedColumn(),
                    '•',
                    ' | '.join(
                        f"{key}: {value[-1]:.4f}" for key, value in self.val_values[epoch_id-1].items()),
                )

            # The test task.
            if task.fields.get("progress_type") == "test":
                self.columns = (
                    f"Test:",
                    rich.progress.BarColumn(),
                    f"{task.completed}/{task.total}",
                    "•",
                    rich.progress.TimeElapsedColumn(),
                    '•',
                    ' | '.join(
                        f"{key}: {value[-1]:.4f}" for key, value in self.test_values.items()),
                )

            yield self.make_tasks_table([task])

    def step_test(self: 'TrainProgress', count: int) -> bool:
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.

        Returns:
            bool: Whether step was successful.
        """
        if len(self.train_tasks) < self.nb_epochs:
            return False

        if self.tasks[self.train_tasks[-1]].completed < self.train_size:
            return False

        if self.val_size > 0:
            if len(self.val_tasks) < self.nb_epochs:
                return False

            if self.tasks[self.val_tasks[-1]].completed < self.val_size:
                return False

        if self.test_size == 0:
            return False

        if self.test_task is None:
            self.test_task = self.add_task(
                "Test",
                progress_type="test",
                total=self.test_size,
            )
            self.update(self.test_task, advance=count)
            self.update(self.total_task, advance=count)
            return True

        if self.test_task is not None:
            self.update(self.test_task, advance=count)
            self.update(self.total_task, advance=count)
            return True

    def step_val(self: 'TrainProgress', count: int) -> bool:
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.

        Returns:
            bool: Whether step was successful.
        """
        if len(self.train_tasks) == 0:
            return False

        if self.tasks[self.train_tasks[-1]].completed < self.train_size:
            return False

        if self.val_size == 0:
            return False

        if len(self.val_tasks) == 0 or (
            len(self.val_tasks) < self.nb_epochs
            and len(self.val_tasks) < len(self.train_tasks)
        ):
            self.val_values.append({})
            self.val_tasks.append(self.add_task(
                f"Val {len(self.val_tasks)+1}",
                progress_type="val",
                epoch_id=len(self.val_tasks)+1,
                total=self.val_size,
            ))
            self.update(self.val_tasks[-1], advance=count)
            self.update(self.total_task, advance=count)
            return True

        if self.tasks[self.val_tasks[-1]].completed < self.val_size:
            self.update(self.val_tasks[-1], advance=count)
            self.update(self.total_task, advance=count)
            return True

    def step_train(self: 'TrainProgress', count: int) -> bool:
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.

        Returns:
            bool: Whether step was successful.
        """
        if len(self.train_tasks) == 0 or self.tasks[self.train_tasks[-1]].completed == self.train_size:
            self.train_values.append({})
            self.train_tasks.append(self.add_task(
                f"Train {len(self.train_tasks)+1}",
                progress_type="train",
                epoch_id=len(self.train_tasks)+1,
                total=self.train_size,
            ))
            self.update(self.train_tasks[-1], advance=count)
            self.update(self.total_task, advance=count)
            return True

        self.update(self.train_tasks[-1], advance=count)
        self.update(self.total_task, advance=count)
        return True

    def step(self: 'TrainProgress', count: int = 1):
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.
        """
        if self.step_test(count):
            return

        if self.step_val(count):
            return

        if self.step_train(count):
            return

        raise RuntimeError("Progress bar already finished.")

    def new_train_values(self: 'TrainProgress', values: dict[str, Any]):
        """Update the progress bar with new values.

        Args:
            values (dict[str, Any]): The new values.
        """
        for key, value in values.items():
            current_value = self.train_values[-1].get(key, [])
            self.train_values[-1][key] = current_value + [value]

    def new_val_values(self: 'TrainProgress', values: dict[str, Any]):
        """Update the progress bar with new values.

        Args:
            values (dict[str, Any]): The new values.
        """
        for key, value in values.items():
            current_value = self.val_values[-1].get(key, [])
            self.val_values[-1][key] = current_value + [value]

    def new_test_values(self: 'TrainProgress', values: dict[str, Any]):
        """Update the progress bar with new values.

        Args:
            values (dict[str, Any]): The new values.
        """
        for key, value in values.items():
            current_value = self.test_values.get(key, [])
            self.test_values[key] = current_value + [value]


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
        val_loader: torch.utils.data.DataLoader | None = None,
        test_loader: torch.utils.data.DataLoader | None = None,
        metrics: List[Callable[[torch.Tensor,
                                torch.Tensor], torch.Tensor]] = [],
    ):
        """Train the model for the given number of epochs.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): The training dataset.
            epochs (int): The number of epochs to train the model for.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            val_loader (torch.utils.data.DataLoader, optional): The validation dataset. Defaults to None.
        """
        with TrainProgress(
            nb_epochs=epochs,
            train_size=len(train_loader),
            val_size=len(val_loader) if val_loader else 0,
            test_size=len(test_loader) if test_loader else 0,
        ) as progress:
            self.progress = progress

            for _ in range(epochs):
                self.train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    metrics,
                )
                if val_loader:
                    self.validate(
                        model,
                        val_loader,
                        metrics + [criterion],
                    )
            if test_loader:
                self.test(
                    model,
                    test_loader,
                    metrics + [criterion],
                )

    def train_epoch(
        self: 'Trainer',
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Train the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): The training dataset.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            metrics (list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        model.train()
        for batch in train_loader:
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
            values = {metric.__name__: metric(output, labels)
                      for metric in metrics}
            values[criterion.__name__] = loss.item()
            self.progress.step()
            self.progress.new_train_values(values)

    def validate(
        self: 'Trainer',
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Validate the model on the given validation dataset.

        Args:
            model (torch.nn.Module): The model to validate.
            val_loader (torch.utils.data.DataLoader): The validation dataset.
            mectrics (list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        model.eval()
        with torch.no_grad():
            metrics_sum = {f'val_{metric.__name__}': 0 for metric in metrics}
            for b_i, batch in enumerate(val_loader):
                inputs = batch[:-1]
                labels = batch[-1]
                output = model(*inputs)
                values = {f'val_{metric.__name__}': metric(output, labels)
                          for metric in metrics}
                for key, value in values.items():
                    metrics_sum[key] += value.item()
                self.progress.step()
                self.progress.new_val_values({
                    key: value / (b_i + 1) for key, value in metrics_sum.items()
                })

    def test(
        self: 'Trainer',
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Test the model on the given test dataset.

        Args:
            model (torch.nn.Module): The model to test.
            test_loader (torch.utils.data.DataLoader): The test dataset.
            mectrics (list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        model.eval()
        with torch.no_grad():
            metrics_sum = {f'test_{metric.__name__}': 0 for metric in metrics}
            for b_i, batch in enumerate(test_loader):
                inputs = batch[:-1]
                labels = batch[-1]
                output = model(*inputs)
                values = {f'test_{metric.__name__}': metric(output, labels)
                          for metric in metrics}
                for key, value in values.items():
                    metrics_sum[key] += value.item()
                self.progress.step()
                self.progress.new_test_values({
                    key: value / (b_i + 1) for key, value in metrics_sum.items()
                })
