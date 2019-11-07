#!/usr/bin/env bash

USAGE="
Usage:
    $0 [OPTION...]

Build and upload Google Cloud AutoML trainer and classifier gears.

Options:
    -h, --help          Print this help and exit
    -T, --skip-train    Skip trainer gear
    -P, --skip-predict  Skip classifier gear
    -u, --upload        Upload gears to Flywheel (using Flywheel CLI)
"

main() {
    local TRAIN_TOGGLE=
    local PREDICT_TOGGLE=
    local UPLOAD_TOGGLE=

    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                printf "$USAGE"; exit 0;;
            -T|--skip-train)
                TRAIN_TOGGLE=false;;
            -P|--skip-predict)
                PREDICT_TOGGLE=false;;
            -u|--upload)
                UPLOAD_TOGGLE=true;;
            *)
                break;;
        esac
        shift
    done

    log "INFO: Build base image ..."
    docker build -t fw-automl-base .

    if [ "$TRAIN_TOGGLE" != false ]; then
        log "INFO: Build trainer gear ..."
        cd train
        docker build -t automl-train .
        if [ "$UPLOAD_TOGGLE" == true ]; then
            log "INFO: Upload trainer gear ..."
            fw gear upload -c analysis
        fi
    fi

    if [ "$PREDICT_TOGGLE" != false ]; then
        log "INFO: Build classifier gear ..."
        cd ../predict
        docker build -t automl-predict .
        if [ "$UPLOAD_TOGGLE" == true ]; then
            log "INFO: Upload classifier gear ..."
            fw gear upload
        fi
    fi
}

log() {
    printf "\n%s\n" "$@" >&2
}

main "$@"
