# hash:sha256:2d631a3fb6c1cd29ad700e2558a133bc7a0285e96d427a33cc7398b9408d90cb
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        jupyter==1.1.1 \
        scanpy==1.11.3 \
        scvi-tools==1.3.3 \
        scikit-misc==0.5.1 \        
    && conda clean -ya


RUN pip install trimesh==4.9.0
RUN pip install --no-cache-dir "numpy<2" "pydeseq2==0.5.2"
# Pin numpyro to match the conda-resolved jax 0.4.31. scvi-tools 1.3.3 otherwise
# pulls numpyro 0.21.0, which requires jax>=0.4.34 (jax.api_util.debug_info) and
# breaks `import scvi`. numpyro is only used by scvi for a few distributions
# (the SCVI model itself runs on torch), so this leaves jax/jaxlib/torch untouched.
RUN pip install --no-cache-dir "numpyro==0.16.1"
# diptest: Hartigan's dip test for the continuum-vs-discrete analysis
# (test_structures.dip_dist_multivariate). Falls back to a pure-scipy Silverman
# test if absent, but diptest gives faster, calibrated p-values.
RUN pip install --no-cache-dir "diptest==0.10.0"
# aind-data-schema: build AIND-standard metadata (data_description/processing) for
# scientist-derived data assets. Used by code/make_mmidas_metadata.py.
RUN pip install --no-cache-dir "aind-data-schema==2.8.1"


COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
