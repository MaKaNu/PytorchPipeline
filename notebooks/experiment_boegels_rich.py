import os
import time

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.rule import Rule

app1_urls = [
    ("https://example.org/app1-part1.tar.gz", 3401321),
    ("https://example.org/app1-part2.tar.gz", 6382217),
]
app2_urls = [
    ("https://example.org/app2-part1.tar.gz", 3929422),
    ("https://example.org/app2-part2.tar.gz", 4156304),
    ("https://example.org/app2-part3.tar.gz", 4356921),
    ("https://example.org/app2-part4.tar.gz", 4582810),
    ("https://example.org/app2-part5.tar.gz", 4731931),
    ("https://example.org/app2-part6.tar.gz", 5103109),
]

step_actions = ("configuring", "building", "installing")


def fake_download(url, tot_size_bytes):
    """Fake download specified URL, in chunks, and show download progress."""
    fn = os.path.basename(url)
    task_id = download_progress.add_task("", filename=fn, total=tot_size_bytes)

    chunk_size = 157 * 1024

    downloaded_bytes = 0
    while downloaded_bytes < tot_size_bytes:
        time.sleep(0.1)
        if downloaded_bytes + chunk_size < tot_size_bytes:
            downloaded_bytes += chunk_size
        else:
            chunk_size = tot_size_bytes - downloaded_bytes
            downloaded_bytes = tot_size_bytes
        download_progress.update(task_id, advance=chunk_size)
    download_progress.stop_task(task_id)
    download_progress.update(task_id, visible=False)


def run_steps(name, step_times, task_id):
    for idx, step_time in enumerate(step_times):
        action = step_actions[idx]
        step_task_id = step_progress_timed.add_task("", action=action, name=name)
        for _ in range(step_time):
            time.sleep(1)
            step_progress_timed.update(step_task_id, advance=1)
        step_progress_timed.stop_task(step_task_id)
        step_progress_timed.update(step_task_id, visible=False)
        task_progress.update(task_id, advance=1)


label_progress = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
)

# progress for a task step that takes a while, but we're not sure how long
step_progress_timed = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("[bold purple]{task.fields[action]}"),
    SpinnerColumn("simpleDots"),
)
# progress for a task step that has a known total target (steps, bytes, ...)
step_progress = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("[bold purple]{task.fields[action]}"),
    BarColumn(),
    TextColumn("({task.completed}/{task.total})"),
)
download_progress = Progress(
    TextColumn("[bold yellow]   Downloading {task.fields[filename]}"),
    BarColumn(),
    DownloadColumn(),
    TransferSpeedColumn(),
    TimeRemainingColumn(),
)
# progress for a single tasks
task_progress = Progress(
    TextColumn("[bold blue]Progress for {task.fields[name]}: {task.percentage:.0f}%"),
    BarColumn(),
    TextColumn("({task.completed} of {task.total} steps done)"),
)
overall_progress = Progress(TimeElapsedColumn(), BarColumn(), TextColumn("{task.description}"))
group = Group(
    label_progress,
    step_progress_timed,
    step_progress,
    download_progress,
    Rule(style="#AAAAAA"),
    task_progress,
    overall_progress,
)
live = Live(group)

apps = [
    ("app 1", app1_urls, (2, 5, 3)),
    ("app 2", app2_urls, (5, 10, 6)),
    ("app 3", [], (1, 5, 3)),
]
overall_task_id = overall_progress.add_task("", total=len(apps))
with live:
    for idx, (name, urls, step_times) in enumerate(apps):
        overall_progress.update(overall_task_id, description=f"[bold #AAAAAA]({idx} out of {len(apps)} apps installed)")
        label_task_id = label_progress.add_task(f"installing {name}")
        task_id = task_progress.add_task("", total=4, name=name)
        download_task_id = step_progress.add_task("", total=len(urls), action="Fetching sources", name=name)
        for url, size_bytes in urls:
            fake_download(url, size_bytes)
            step_progress.update(download_task_id, advance=1)
        step_progress.update(download_task_id, visible=False)
        task_progress.update(task_id, advance=1)
        run_steps(name, step_times, task_id)
        task_progress.update(task_id, visible=False)
        label_progress.stop_task(label_task_id)
        label_progress.update(label_task_id, description=f"[bold green]{name} installed!")
        overall_progress.update(overall_task_id, advance=1)

    overall_progress.update(overall_task_id, description=f"[bold green]{len(apps)} apps installed, done!")
