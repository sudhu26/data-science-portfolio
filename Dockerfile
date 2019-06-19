FROM python:3.6.8-stretch

RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    bash \
    curl \
    git \
    nano \
    net-tools \
    software-properties-common \
    ssh \
    sudo \
    tar \
    tree \
    wget
RUN  apt-get clean

WORKDIR /requirements
COPY requirements.txt /requirements

RUN pip install pip --upgrade
RUN pip install -r requirements.txt

# how do you get mlmachine and quickplot in here?

# RUN pip uninstall tornado -y
# RUN pip install tornado==5.1.1

# # activate notebook extensions
# RUN jupyter contrib nbextension install --sys-prefix
# RUN jupyter nbextension enable snippets/main
# RUN jupyter nbextension enable toc2/main
# RUN jupyter nbextension enable highlight_selected_word/main
# RUN jupyter nbextension enable codefolding/main
# RUN jupyter nbextension enable execute_time/ExecuteTime
# RUN jupyter nbextension enable livemdpreview/livemdpreview
# RUN jupyter nbextension enable snippets_menu/main
# RUN jupyter nbextension enable toggle_all_line_numbers/main
# RUN jupyter nbextension enable collapsible_headings/main

# # Set jupyter theme
# RUN jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T -dfs 8

#
ADD . /home
WORKDIR /home


# CMD jupyter notebook --no-browser --ip 0.0.0.0 --allow-root --port 8888
