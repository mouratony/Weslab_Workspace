FROM condaforge/miniforge3

WORKDIR /weslab_workspace

COPY environment.yml .

RUN conda env create -f environment.yml

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "wesl_tony_env"]
CMD ["zsh"]
