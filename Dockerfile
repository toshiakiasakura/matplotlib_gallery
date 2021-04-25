FROM jupyter/scipy-notebook:584f43f06586

WORKDIR /workdir
EXPOSE 8888

RUN pip install contextplt
# sphinx setting
RUN conda install sphinx -y && \
    pip install sphinx-autodoc-typehints && \
    pip install jupyterlab_vim

