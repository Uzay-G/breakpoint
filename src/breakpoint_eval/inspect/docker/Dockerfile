FROM python:3.11@sha256:ce3b954c9285a7a145cba620bae03db836ab890b6b9e0d05a3ca522ea00dfbc9

# Install some basic inspect dependencies
RUN apt-get update && apt-get install -y pipx && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pipx ensurepath
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install inspect-tool-support && inspect-tool-support post-install

# Create a virtual environment and set it to be used by default
ENV VIRTUAL_ENV=/root/.venv
RUN python -m venv $VIRTUAL_ENV && /root/.venv/bin/pip install --no-cache-dir --upgrade pip==25.1.1
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install shared dependencies
RUN pip install --no-cache-dir pytest==8.4.1 pytest-reportlog==0.4.0

# Make path work in login shells
RUN echo "PATH=\"$VIRTUAL_ENV/bin:$PATH:/root/.local/bin\"" >> /root/.bashrc

# Clone repo
ARG REPO_URL
ARG REPO_COMMIT_HASH
RUN git clone ${REPO_URL} /root/code && \
    cd /root/code && \
    git checkout ${REPO_COMMIT_HASH}

# Set the working directory to the cloned repository
WORKDIR /root/code

# Patch and reset git history
ARG DIFF_BASE64
RUN echo "${DIFF_BASE64}" | base64 -d | git apply && \
    rm -rf .git && \
    git config --global user.email "eval-user@example.org" && \
    git config --global user.name "eval-user" && \
    git init && git add . && git commit -m "Set up repo for evaluation."

# Install repo dependencies
# We do this after the patch to ensure that the pre-patch data cannot accidentally be present on the system
# due to some pip install shenanigans. This is to make cheating more difficult.
RUN pip install --no-cache-dir -e . && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi && \
    if [ -f requirements/requirements_dev.txt ]; then pip install --no-cache-dir -r requirements/requirements_dev.txt; fi
