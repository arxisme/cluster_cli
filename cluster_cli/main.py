import os
import subprocess
import yaml
import json
import textwrap
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from fabric import Connection

app = typer.Typer(help="Personal cluster orchestrator.")
console = Console()

@app.command()
def init(project_name: str = typer.Argument(..., help="Name of the new cluster project")):
    """Scaffold a new cluster project."""
    base_dir = Path(project_name)
    
    if base_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Directory '{project_name}' already exists.")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Scaffolding project:[/bold cyan] {project_name}...")

    # 1. Create the directory tree
    directories = [
        base_dir / ".cluster",   
        base_dir / "src",        
        base_dir / "scripts",    
        base_dir / "datasets",   
        base_dir / "models",     
        base_dir / "results",    
    ]
    
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Generate cluster.yaml (Tailored to CSE Cluster)
    yaml_content = textwrap.dedent(f"""\
        project_name: {project_name}

        cluster:
          host: 172.16.112.202
          user: arshad.ahmed  # Defaulting to your username

        stages:
          prepare:
            - python3 scripts/download.py
          
          build:
            base_image: pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
            auto_detect_dependencies: true

          run:
            partition: gpu-P100
            gpus: 1
            cpus: 8
            memory: 32G
            time: "04:00:00"
            command: python3 src/train.py --output ./results/
    """)
    (base_dir / "cluster.yaml").write_text(yaml_content)

    # 3. Generate stub files
    stub_train = textwrap.dedent("""\
        import os
        print("Training initiated...")
        os.makedirs("./results", exist_ok=True)
        with open("./results/model_weights.pth", "w") as f:
            f.write("Epoch 100: Loss 0.001 - Target reached.")
        print("Training completed. Results saved.")
    """)
    (base_dir / "src" / "train.py").write_text(stub_train)
    (base_dir / "scripts" / "download.py").write_text('print("Downloading models/datasets...")\n')
    (base_dir / "requirements.txt").write_text("torch\nnumpy\n")

    # 4. Success Output
    console.print(f"[bold green]✓ Project initialized successfully![/bold green]")
    console.print(f"Next steps:\n  1. [bold]cd {project_name}[/bold]\n  2. Run [bold]cluster build[/bold]")


@app.command()
def build():
    """Dynamically generate Apptainer def and build the image."""
    base_dir = Path.cwd()
    yaml_path = base_dir / "cluster.yaml"
    req_path = base_dir / "requirements.txt"
    src_dir = base_dir / "src"

    if not yaml_path.exists():
        console.print("[bold red]Error:[/bold red] cluster.yaml not found. Are you in the project root?")
        raise typer.Exit(code=1)

    # 1. Parse the YAML config
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    project_name = config.get("project_name", "cluster_project")
    base_image = config["stages"]["build"]["base_image"].replace("docker://", "")
    auto_detect = config["stages"]["build"].get("auto_detect_dependencies", False)
    image_name = f"{project_name}.sif"

    # 2. Auto-Detect Dependencies
    if auto_detect and src_dir.exists():
        console.print(f"[bold cyan]Auto-detecting Python dependencies in src/ ...[/bold cyan]")
        try:
            subprocess.run(
                ["pipreqs", str(src_dir), "--savepath", str(req_path), "--force"],
                check=True,
                capture_output=True
            )
            console.print("[bold green]✓ requirements.txt generated.[/bold green]")
        except subprocess.CalledProcessError:
            console.print("[bold yellow]Warning:[/bold yellow] pipreqs failed. Proceeding with existing requirements.txt.")
        except FileNotFoundError:
            console.print("[bold yellow]Warning:[/bold yellow] 'pipreqs' not found locally. Install with 'pip install pipreqs'.")

    # 3. Read Python dependencies
    reqs = ""
    if req_path.exists():
        reqs = " ".join([line.strip() for line in req_path.read_text().splitlines() if line.strip()])

    console.print(f"[bold cyan]Generating apptainer.def for:[/bold cyan] {project_name}...")

    # 4. Construct the definition file
    def_content = textwrap.dedent(f"""\
        Bootstrap: docker
        From: {base_image}

        %post
            apt-get update
            apt-get install -y python3 python3-pip git wget
            
            if [ -n "{reqs}" ]; then
                pip3 install {reqs}
            fi
            
            apt-get clean
            rm -rf /var/lib/apt/lists/*

        %environment
            export LC_ALL=C
            export PATH=/usr/local/bin:$PATH
    """)
    
    def_path = base_dir / "apptainer.def"
    def_path.write_text(def_content)
    console.print("[bold green]✓[/bold green] apptainer.def generated.")

    # 5. Execute the Apptainer build
    console.print(f"[bold yellow]Building {image_name} (this will take a while)...[/bold yellow]")
    try:
        subprocess.run(
            ["apptainer", "build", "--fakeroot", "--force", image_name, "apptainer.def"], 
            check=True
        )
        console.print(f"\n[bold green]✓ Successfully built container:[/bold green] {image_name}")
    except subprocess.CalledProcessError:
        console.print("\n[bold red]Error:[/bold red] Apptainer build process failed.")
        raise typer.Exit(code=1)


@app.command()
def run(ask_pass: bool = typer.Option(False, "--ask-pass", "-p", help="Prompt for SSH password")):
    """Execute the pipeline defined in cluster.yaml."""
    base_dir = Path.cwd()
    yaml_path = base_dir / "cluster.yaml"
    state_dir = base_dir / ".cluster"
    
    if not yaml_path.exists():
        console.print("[bold red]Error:[/bold red] cluster.yaml not found.")
        raise typer.Exit(code=1)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    project_name = config.get("project_name", "cluster_project")
    user = config["cluster"].get("user")
    host = config["cluster"].get("host", "172.16.112.202")

    if not user or user == "null":
        console.print("[bold red]Error:[/bold red] Please update cluster.yaml with your cluster username.")
        raise typer.Exit(code=1)

    image_name = f"{project_name}.sif"

    # 1. Local Prepare Stage
    console.print("[bold cyan]Running prepare stage...[/bold cyan]")
    for cmd in config["stages"].get("prepare", []):
        console.print(f"  Executing: [dim]{cmd}[/dim]")
        subprocess.run(cmd, shell=True, check=True)

    # 2. SSH Connection and Transfer
    console.print(f"\n[bold cyan]Connecting to {host}...[/bold cyan]")
    
    connect_kwargs = {}
    if ask_pass:
        password = typer.prompt(f"Enter SSH password for {user}@{host}", hide_input=True)
        connect_kwargs["password"] = password

    try:
        with Connection(host=host, user=user, connect_kwargs=connect_kwargs) as c:
            c.run(f"mkdir -p ~/projects/{project_name}/results", hide=True)
            c.run(f"mkdir -p ~/projects/{project_name}/src", hide=True)
            c.run("mkdir -p ~/containers", hide=True)

            if (base_dir / image_name).exists():
                console.print(f"[bold yellow]Transferring {image_name}...[/bold yellow]")
                c.put(str(base_dir / image_name), f"containers/{image_name}")
            else:
                console.print(f"[bold red]Error:[/bold red] {image_name} not found. Run 'cluster build' first.")
                raise typer.Exit(code=1)

            console.print("[bold yellow]Syncing workspace (src, models, datasets)...[/bold yellow]")
            import subprocess
            subprocess.run(["tar", "-czf", "sync.tar.gz", "src", "models", "datasets"], check=True)
            c.put(str(base_dir / "sync.tar.gz"), f"projects/{project_name}/sync.tar.gz")
            c.run(f"cd projects/{project_name} && tar -xzf sync.tar.gz && rm sync.tar.gz", hide=True)
            (base_dir / "sync.tar.gz").unlink()

            # 3. Generate SLURM Script
            console.print("[bold cyan]Generating SLURM script...[/bold cyan]")
            run_cfg = config["stages"]["run"]
            
            slurm_script = textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={project_name}
                #SBATCH --output=%j.out
                #SBATCH --error=%j.err
                #SBATCH --time={run_cfg.get('time', '04:00:00')}
                #SBATCH --partition={run_cfg.get('partition', 'gpu-P100')}
                #SBATCH --gres=gpu:{run_cfg.get('gpus', 1)}
                #SBATCH --cpus-per-task={run_cfg.get('cpus', 8)}
                #SBATCH --mem={run_cfg.get('memory', '32G')}

                cd "$TMPDIR" || exit 1
                echo "Job $SLURM_JOB_ID running on $(hostname)"
                
                cp -r $HOME/projects/{project_name}/src ./
                cp -r $HOME/projects/{project_name}/models ./
                cp -r $HOME/projects/{project_name}/datasets ./
                cp $HOME/containers/{image_name} ./
                mkdir -p ./results

                apptainer exec --nv ./{image_name} {run_cfg.get('command')}
                
                cp -r ./results/* $HOME/projects/{project_name}/results/ || true
                echo "Job completed at $(date)"
            """)
            
            script_path = base_dir / "submit.sh"
            script_path.write_text(slurm_script)
            c.put(str(script_path), f"projects/{project_name}/submit.sh")

            # 4. Submit Job to SLURM
            console.print("[bold cyan]Submitting job to SLURM...[/bold cyan]")
            result = c.run(f"cd projects/{project_name} && sbatch submit.sh", hide=True)
            
            output = result.stdout.strip()
            job_id = output.split()[-1]
            console.print(f"[bold green]✓ Job submitted successfully! Job ID:[/bold green] {job_id}")

            state_file = state_dir / "state.json"
            state_file.write_text(json.dumps({"current_job_id": job_id}))

    except Exception as e:
        console.print(f"\n[bold red]SSH/Transfer Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def status(ask_pass: bool = typer.Option(False, "--ask-pass", "-p", help="Prompt for SSH password")):
    """Check the status of the current job."""
    base_dir = Path.cwd()
    yaml_path = base_dir / "cluster.yaml"
    state_path = base_dir / ".cluster" / "state.json"
    
    if not yaml_path.exists() or not state_path.exists():
        console.print("[bold red]Error:[/bold red] Project not found or no active job state. Run 'cluster run' first.")
        raise typer.Exit(code=1)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    with open(state_path, "r") as f:
        state = json.load(f)

    project_name = config.get("project_name", "Unknown")
    user = config["cluster"].get("user")
    host = config["cluster"].get("host", "172.16.112.202")
    job_id = state.get("current_job_id")

    connect_kwargs = {}
    if ask_pass:
        password = typer.prompt(f"Enter SSH password for {user}@{host}", hide_input=True)
        connect_kwargs["password"] = password

    console.print(f"[bold cyan]Fetching status for Job {job_id}...[/bold cyan]")

    try:
        with Connection(host=host, user=user, connect_kwargs=connect_kwargs) as c:
            result = c.run(f"scontrol show job {job_id}", hide=True, warn=True)
            
            job_state, run_time, nodes, partition = "UNKNOWN", "N/A", "N/A", "N/A"

            if result.ok:
                output = result.stdout
                state_match = re.search(r"JobState=(\w+)", output)
                time_match = re.search(r"RunTime=([^\s]+)", output)
                node_match = re.search(r"NodeList=([^\s]+)", output)
                part_match = re.search(r"Partition=([^\s]+)", output)

                job_state = state_match.group(1) if state_match else "UNKNOWN"
                run_time = time_match.group(1) if time_match else "00:00:00"
                nodes = node_match.group(1) if node_match and node_match.group(1) != "(null)" else "Queued"
                partition = part_match.group(1) if part_match else "UNKNOWN"
            else:
                sacct_res = c.run(f"sacct -j {job_id} --format=State,Elapsed,Partition -P -n", hide=True, warn=True)
                if sacct_res.ok and sacct_res.stdout.strip():
                    parts = sacct_res.stdout.strip().split('\n')[0].split('|')
                    if len(parts) >= 3:
                        job_state = parts[0]
                        run_time = parts[1]
                        partition = parts[2]
                        nodes = "Released"

            state_color = "yellow"
            if job_state in ["RUNNING", "COMPLETED"]: state_color = "green"
            elif job_state in ["FAILED", "CANCELLED", "TIMEOUT"]: state_color = "red"

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Property", style="cyan", justify="right")
            table.add_column("Value", style="white")

            table.add_row("Project:", project_name)
            table.add_row("Job ID:", str(job_id))
            table.add_row("Partition:", partition)
            table.add_row("Status:", f"[bold {state_color}]{job_state}[/bold {state_color}]")
            table.add_row("Elapsed Time:", run_time)
            table.add_row("Compute Node:", nodes)

            panel = Panel(Align.center(table), title="[bold blue]Cluster Status[/bold blue]", border_style="blue", expand=False)
            console.print("\n", panel, "\n")

    except Exception as e:
        console.print(f"\n[bold red]SSH/Status Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def logs(
    ask_pass: bool = typer.Option(False, "--ask-pass", "-p", help="Prompt for SSH password"),
    error_only: bool = typer.Option(False, "--error", "-e", help="Show only the error log")
):
    """Fetch and display the logs for the current job."""
    base_dir = Path.cwd()
    yaml_path = base_dir / "cluster.yaml"
    state_path = base_dir / ".cluster" / "state.json"
    
    if not yaml_path.exists() or not state_path.exists():
        console.print("[bold red]Error:[/bold red] Project not found or no active job state.")
        raise typer.Exit(code=1)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    with open(state_path, "r") as f:
        state = json.load(f)

    project_name = config.get("project_name", "Unknown")
    user = config["cluster"].get("user")
    host = config["cluster"].get("host", "172.16.112.202")
    job_id = state.get("current_job_id")

    connect_kwargs = {}
    if ask_pass:
        password = typer.prompt(f"Enter SSH password for {user}@{host}", hide_input=True)
        connect_kwargs["password"] = password

    console.print(f"[bold cyan]Fetching logs for Job {job_id}...[/bold cyan]\n")

    try:
        with Connection(host=host, user=user, connect_kwargs=connect_kwargs) as c:
            out_file = f"projects/{project_name}/{job_id}.out"
            err_file = f"projects/{project_name}/{job_id}.err"

            err_result = c.run(f"cat {err_file}", hide=True, warn=True)
            if err_result.ok and err_result.stdout.strip():
                console.print(Panel(err_result.stdout.strip(), title=f"[bold red]STDERR ({job_id}.err)[/bold red]", border_style="red"))
            elif err_result.ok:
                if error_only: console.print("[dim italic]Error log is empty.[/dim italic]")
            
            if not error_only:
                out_result = c.run(f"cat {out_file}", hide=True, warn=True)
                if out_result.ok and out_result.stdout.strip():
                    console.print(Panel(out_result.stdout.strip(), title=f"[bold green]STDOUT ({job_id}.out)[/bold green]", border_style="green"))
                elif out_result.ok:
                    console.print("[dim italic]Standard output is empty (job might still be pending).[/dim italic]")

    except Exception as e:
        console.print(f"\n[bold red]SSH/Log Error:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def pull(ask_pass: bool = typer.Option(False, "--ask-pass", "-p", help="Prompt for SSH password")):
    """Pull the results folder from the cluster to your local machine."""
    base_dir = Path.cwd()
    yaml_path = base_dir / "cluster.yaml"
    
    if not yaml_path.exists():
        console.print("[bold red]Error:[/bold red] cluster.yaml not found.")
        raise typer.Exit(code=1)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    project_name = config.get("project_name", "Unknown")
    user = config["cluster"].get("user")
    host = config["cluster"].get("host", "172.16.112.202")

    connect_kwargs = {}
    if ask_pass:
        password = typer.prompt(f"Enter SSH password for {user}@{host}", hide_input=True)
        connect_kwargs["password"] = password

    console.print(f"[bold cyan]Pulling results for {project_name}...[/bold cyan]")

    try:
        with Connection(host=host, user=user, connect_kwargs=connect_kwargs) as c:
            remote_dir = f"projects/{project_name}/results"
            remote_tar = f"projects/{project_name}/results_sync.tar.gz"
            local_tar = base_dir / "results_sync.tar.gz"

            # 1. Verify remote directory exists
            check = c.run(f"test -d {remote_dir}", warn=True, hide=True)
            if check.failed:
                console.print("[bold yellow]No results folder found on the cluster yet.[/bold yellow]")
                raise typer.Exit()

            # 2. Archive remote files
            console.print("  [dim]Archiving remote files...[/dim]")
            c.run(f"cd projects/{project_name} && tar -czf results_sync.tar.gz results/", hide=True)

            # 3. Download the archive
            console.print("  [dim]Downloading to local machine...[/dim]")
            c.get(remote_tar, str(local_tar))

            # 4. Extract locally (this safely merges with your local folder)
            console.print("  [dim]Extracting files...[/dim]")
            subprocess.run(["tar", "-xzf", str(local_tar)], check=True)

            # 5. Clean up temporary archives
            c.run(f"rm {remote_tar}", hide=True)
            if local_tar.exists():
                local_tar.unlink()

            console.print("[bold green]✓ Results successfully synced to ./results/[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]SSH/Transfer Error:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()