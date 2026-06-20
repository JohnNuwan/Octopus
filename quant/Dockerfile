FROM julia:1.11-bullseye AS quant

WORKDIR /app

COPY Project.toml Manifest.toml ./
RUN julia -e 'import Pkg; Pkg.instantiate()'

COPY src/ ./src/

CMD ["julia", "--project=.", "-e", "import Octopus; Octopus.run()"]