#!/bin/bash
file="$1"

key_id="AKIA26D655VR2TW56VHH"
key_secret="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"

BLOB_NAMES=(
    datasets_fullband/clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2
    datasets_fullband/clean_fullband/datasets_fullband.clean_fullband.russian_speech_001_4.31_NA.tar.bz2

    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2
    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2
    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2
    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2
    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2
    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2

    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
    datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2
    datasets/datasets.impulse_responses.tar.bz2
)

AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive"

#path="some-directory/$file"
bucket="ai-thesis"
content_type="application/octet-stream"
date="$(LC_ALL=C date -u +"%a, %d %b %Y %X %z")"
#md5="$(openssl md5 -binary < "$file" | base64)"

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"


    sig="$(printf "PUT\n$content_type\n$date\n/$bucket/$BLOB" | openssl sha1 -binary -hmac "$key_secret" | base64)"

#    curl "$URL" | tar -C "$OUTPUT_PATH" -f - -x -j

    curl -OL "$URL" | tar -C http://$bucket.s3.amazonaws.com/$BLOB -f - -x -j \
        -H "Date: $date" \
        -H "Authorization: AWS $key_id:$sig" \
        -H "Content-Type: $content_type" \
#        -H "Content-MD5: $md5"
done
