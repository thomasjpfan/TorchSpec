import modal

app = modal.App("torch-spec")
SHA = "985f90d56d8d022cfce48b3b8da9ab181b144b1a"
HF_HOME = "/mnt/hf_home"

image = (
    modal.Image.debian_slim("3.12")
    .apt_install(
        "git",
        "libcurl4",
        "libcurl4-openssl-dev",
        "ibverbs-providers",
        "libibverbs1",
        "librdmacm1",
        "libnuma1",
        "liburing2",
        "libyaml-0-2",
    )
    .run_commands(
        "git clone https://github.com/torchspec-project/TorchSpec --depth 1",
        f"cd TorchSpec && git fetch --depth 1 origin {SHA} && git checkout {SHA}",
    )
    .workdir("TorchSpec")
    .run_commands("./tools/build_conda.sh current sglang")
    .env({"HF_HOME": HF_HOME})
    .add_local_file(
        "examples/qwen3-8b-single-node/run.sh", "/TorchSpec/examples/qwen3-8b-single-node/run.sh"
    )
)

hf_volume = modal.Volume.from_name("hf-home", create_if_missing=True)


@app.function(
    image=image,
    volumes={HF_HOME: hf_volume},
    gpu="A100:4",
    timeout=60 * 60,
)
def spec_it():
    import subprocess

    subprocess.run(["./examples/qwen3-8b-single-node/run.sh"])


if __name__ == "__main__":
    func = modal.Function.from_name("torch-spec", "spec_it")
    func.spawn()
