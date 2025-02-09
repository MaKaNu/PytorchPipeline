import time
from functools import wraps

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


def rgb2hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


active_bars = list(range(10))

N = len(active_bars)
colors = [
    (
        # Red channel
        int(255 * (2 * t if t <= 0.5 else 1)),
        # Green channel
        int(255 * (2 * t if t <= 0.5 else 1 - 2 * (t - 0.5))),
        # Blue channel
        int(255 * (1 if t <= 0.5 else 1 - 2 * (t - 0.5))),
    )
    for i in range(N)
    for t in [i / (N - 1)]
]


def too_many():  # noqa: C901
    with Progress() as progress:
        task1 = progress.add_task("[red]Downloading...", total=2)
        task2 = progress.add_task("[green]Processing...", total=2)
        task3 = progress.add_task("[cyan]Cooking...", total=2)
        task4 = progress.add_task("[magenta]Baking...", total=2)
        task5 = progress.add_task("[yellow]Eating...", total=2)
        task6 = progress.add_task("[blue]Digesting...", total=2)
        task7 = progress.add_task("[white]Sleeping...", total=2)
        task8 = progress.add_task("[orange]Dreaming...", total=2)
        task9 = progress.add_task("[purple]Waking up...", total=2)
        task10 = progress.add_task("[black]Starting over...", total=5)

        while not progress.finished:
            for i in range(2):
                progress.update(task1, completed=i)
                for j in range(2):
                    progress.update(task2, completed=j)
                    for k in range(2):
                        progress.update(task3, completed=k)
                        for l in range(2):  # noqa: E741
                            progress.update(task4, completed=l)
                            for m in range(2):
                                progress.update(task5, completed=m)
                                for n in range(2):
                                    progress.update(task6, completed=n)
                                    for o in range(2):
                                        progress.update(task7, completed=o)
                                        for p in range(2):
                                            progress.update(task8, completed=p)
                                            for q in range(2):
                                                progress.update(task9, completed=q)
                                                for r in range(5):
                                                    progress.update(task10, completed=r)
                                                    time.sleep(0.01)
                                                else:
                                                    progress.advance(task10)
                                            else:
                                                progress.advance(task9)
                                        else:
                                            progress.advance(task8)
                                    else:
                                        progress.advance(task7)
                                else:
                                    progress.advance(task6)
                            else:
                                progress.advance(task5)
                        else:
                            progress.advance(task4)
                    else:
                        progress.advance(task3)
                else:
                    progress.advance(task2)
            else:
                progress.advance(task1)


def simple_nested():
    with Progress() as progress:
        task1 = progress.add_task("[red]Downloading...", total=4)
        task2 = progress.add_task("[green]Processing...", total=4)
        task3 = progress.add_task("[cyan]Cooking...", total=200)

        for _ in range(4):
            progress.remove_task(task2)
            task2 = progress.add_task("[green]Processing...", total=4)
            for _ in range(4):
                progress.remove_task(task3)
                task3 = progress.add_task("[cyan]Cooking...", total=200)
                for _ in range(200):
                    progress.update(task3, advance=1)
                    time.sleep(0.01)
                else:
                    progress.update(task2, advance=1)
            else:
                progress.update(task1, advance=1)


def conditional():
    # Global setting to toggle behavior
    USE_PROGRESS = True  # Set this to False to disable progress tracking

    custom_bar = BarColumn(style="#333333", complete_style="{task.fields[color]}", finished_style="#00ff00")

    def conditional_progress(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if USE_PROGRESS:
                with Progress(
                    "SAM2SEGNET",
                    TextColumn(
                        "[bold blue][progress.description]{task.description}: {task.fields[process_status]}",
                        justify="right",
                    ),
                    custom_bar,
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    "•",
                    TimeRemainingColumn(),
                ) as progress:
                    return func(progress, *args, **kwargs)
            else:
                return func(None, *args, **kwargs)  # No progress manager

        return wrapper

    @conditional_progress
    def my_task(progress):
        task1 = None
        task2 = None
        if progress:
            task1 = progress.add_task("Process1", total=5, process_status="IDLE", color="#FFFF00")
            task2 = progress.add_task("Process2", total=5, process_status="IDLE", color="#00FFFF")

        loss = 0.0
        for _ in range(5):
            loss += 0.1
            mean_loss = loss / 5
            if progress:
                process_status = f"LOSS: {loss:.4f}, MEAN LOSS: {mean_loss:.4f}"
                progress.update(task1, process_status=process_status)
                progress.advance(task1)
            # Do other work
            time.sleep(0.5)
        for _ in range(5):
            loss += 0.1
            mean_loss = loss / 5
            if progress:
                process_status = f"LOSS: {loss:.4f}, MEAN LOSS: {mean_loss:.4f}"
                progress.update(task2, process_status=process_status)
                progress.advance(task2)
            # Do other work
            time.sleep(0.5)

    # Run the function
    my_task()


def multi_nested():  # noqa: C901
    class ProgressManager:
        def __init__(self):
            self.progress = Progress(
                TextColumn("[bold #FFF987]MyProgress"),
                BarColumn(complete_style="#FFF987"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
            )

    # Define the decorator
    def progress_task(progress, task_name):
        def decorator(func):
            @wraps(func)
            def wrapper(total, *args, **kwargs):
                # Add task to progress
                task_id = progress.add_task(task_name, total=total)

                # Call the function with task_id
                result = func(task_id, total, progress, *args, **kwargs)

                # Hide task when done
                progress.update(task_id, visible=False)
                return result

            return wrapper

        return decorator

    progresses = {
        "epoch": {"total": 2, "color": "#ff0000"},
        "train": {"total": 40, "color": "#00ff00"},
        "val": {"total": 40, "color": "#0000ff"},
    }

    grouped = {}
    for progress in progresses:
        grouped[progress] = Progress(
            TextColumn(f"[bold {progresses[progress]['color']}]{progress}"),
            BarColumn(complete_style=f"{progresses[progress]['color']}"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
        )

    group = Group(*grouped.values())

    live = Live(group)

    @progress_task(grouped["train"], "train")
    def run_train(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    @progress_task(grouped["val"], "val")
    def run_val(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    @progress_task(grouped["epoch"], "epoch")
    def run_epoch(epoch_id, total, progress):
        for _ in range(total):
            run_train(40)
            run_val(40)
            progress.advance(epoch_id)

    with live:
        run_epoch(2)


def decorated():
    progress = Progress(
        TextColumn(f"[bold {'#FFF987'}]{'MyProgress'}"),
        BarColumn(complete_style="#FFF987"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
    )

    group = Group(progress)

    live = Live(group)

    # Define the decorator
    def progress_task(progress):
        def decorator(func):
            @wraps(func)
            def wrapper(task_name, total, *args, **kwargs):
                # Add task to progress
                task_id = progress.add_task(task_name, total=total)

                # Call the function with task_id
                result = func(task_id, task_name, total, *args, **kwargs)

                # Hide task when done
                progress.update(task_id, visible=False)
                return result

            return wrapper

        return decorator

    # Use the decorator
    @progress_task(progress)
    def do_something(task_id, task_name, total):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    # Run inside Live context
    with live:
        do_something("DO SOMETHING", 40)


def class_based():  # noqa: C901
    class ProgressManager:
        def __init__(self):
            epoch_progress = Progress(
                TextColumn("[bold #FF2255]epoch"),
                BarColumn(style="#333333", complete_style="#FF2255", finished_style="#22FF55"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
            )
            train_progress = Progress(
                TextColumn("[bold #5522FF]{task.description}"),
                BarColumn(style="#333333", complete_style="#5522FF", finished_style="#22FF55"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
            )

            self.progress_dict = {
                "epoch": epoch_progress,
                "train_val": train_progress,
            }
            group = Group(*self.progress_dict.values())

            self.live = Live(group)

        def progress_task(self, task_name, visible=True):
            def decorator(func):
                @wraps(func)
                def wrapper(total, *args, **kwargs):
                    progress_key = next((key for key in self.progress_dict if task_name in key), None)
                    progress = self.progress_dict[progress_key]
                    # Add task to progress
                    task_id = progress.add_task(task_name, total=total)

                    # Call the function with task_id
                    result = func(task_id, total, progress, *args, **kwargs)

                    # Hide task when done
                    progress.update(task_id, visible=visible)
                    return result

                return wrapper

            return decorator

    progress_manager = ProgressManager()

    @progress_manager.progress_task("train", visible=False)
    def run_train(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    @progress_manager.progress_task("val", visible=False)
    def run_val(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    @progress_manager.progress_task("epoch")
    def run_epoch(epoch_id, total, progress):
        for _ in range(total):
            run_train(40)
            run_val(40)
            progress.advance(epoch_id)

    with progress_manager.live:
        run_epoch(2)


def class_based2():  # noqa: C901
    class ProgressManager2:
        def __init__(self, console=None):
            self.console = console
            self.progress_dict = {
                "epoch": self._create_progress("#FF2255"),
                "train_val": self._create_progress("#5522FF"),
            }
            group = Group(*self.progress_dict.values())

            self.live = Live(group)

        def _create_progress(self, color="#F55500"):
            return Progress(
                TextColumn(f"[bold{color}]" + "{task.description}"),
                BarColumn(style="#333333", complete_style=color, finished_style="#22FF55"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                console=self.console,
            )

        def progress_task(self, task_name, visible=True):
            def decorator(func):
                @wraps(func)
                def wrapper(total, *args, **kwargs):
                    progress_key = next((key for key in self.progress_dict if task_name.lower() in key), None)
                    progress = self.progress_dict[progress_key]
                    # Add task to progress
                    task_id = progress.add_task(task_name, total=total)

                    # Call the function with task_id
                    result = func(task_id, total, progress, *args, **kwargs)

                    # Hide task when done
                    progress.update(task_id, visible=visible)
                    return result

                return wrapper

            return decorator

    console = Console(force_terminal=True)
    progress_manager = ProgressManager2(console)

    @progress_manager.progress_task("train", visible=False)
    def run_train(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    @progress_manager.progress_task("val", visible=False)
    def run_val(task_id, total, progress):
        for _ in range(total):
            progress.advance(task_id)
            time.sleep(0.1)

    @progress_manager.progress_task("Epoch")
    def run_epoch(epoch_id, total, progress):
        for _ in range(total):
            run_train(40)
            run_val(40)
            progress.advance(epoch_id)

    with progress_manager.live:
        run_epoch(2)


class_based2()
