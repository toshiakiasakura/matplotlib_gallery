FROM jupyter/datascience-notebook:lab-2.2.9

WORKDIR /workdir
EXPOSE 8888

# jupyter lab extensions. 
RUN conda install -c conda-forge jupyterlab-snippets && \
    conda install -c conda-forge jupyterlab-git -y && \
    jupyter labextension install jupyterlab-plotly@4.14.3 --no-build && \
    jupyter labextension install @axlair/jupyterlab_vim --no-build && \
    jupyter lab build

# python package installation. 
RUN conda install -c conda-forge plotly -y && \
    pip install japanize-matplotlib && \
    pip install ipynb_path && \
    pip install mojimoji && \
    pip install Levenshtein



# install vim 
#RUN apt-get update && \
#    apt-get install -y vim

#RUN sudo echo "alias jpt_lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root '" >> /root/.bashrc

