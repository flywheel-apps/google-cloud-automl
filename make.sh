#!/usr/bin/env bash
USAGE="
Usage:
    $0 [OPTION...]

Build and upload Google Cloud AutoML train and prediction gears.

Options:
    -h, --help          Print this help and exit
    -T, --skip-train    Skip automl-train gear
    -P, --skip-predict  Skip automl-predict gear
    -u, --upload VER    Upload gear VER using cli
"


main() {
    [ "$DEBUG" ] && set -o xtrace
    set -o nounset
    set -o pipefail

    local BASEDIR=$(cd $(dirname $0) && pwd)
    local SKIP_TRAIN=false
    local SKIP_PREDICT=false
    local VERSION=

    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                printf "$USAGE"; exit 0;;
            -T|--skip-train)
                SKIP_TRAIN=true;;
            -P|--skip-predict)
                SKIP_PREDICT=true;;
            -u|--upload)
                VERSION=$2; shift;;
            *)
                break;;
        esac
        shift
    done

    log "INFO: Build base image"
    cd "$BASEDIR"
    docker build -t fw-automl-base .

    if [ "$SKIP_TRAIN" != true ]; then
        cd "$BASEDIR/train"
        log "INFO: Building automl-train"
        docker build -t automl-train .
        if [ -n "$VERSION" ]; then
            log "INFO: Uploading automl-train $VERSION"
            sed -Ei "s/^    \"version\":.*\$/    \"version\": \"$VERSION\",/" manifest.json
            fw gear upload
        fi
    fi

    if [ "$SKIP_PREDICT" != true ]; then
        cd "$BASEDIR/predict"
        log "INFO: Building automl-predict"
        docker build -t automl-predict .
        if [ -n "$VERSION" ]; then
            log "INFO: Uploading automl-predict $VERSION"
            sed -Ei "s/^    \"version\":.*\$/    \"version\": \"$VERSION\",/" manifest.json
            fw gear upload
        fi
    fi
}


log() {
    printf "%s\n" "$*" >&2
}


main "$@"
