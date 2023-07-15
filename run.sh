#!/bin/bash

function run() {
  echo $command
  echo ${args}
  ${command}
}

config_path=$1

command="python -u main_fedavg_AFCL.py"

while IFS="=" read -r arg value; do

  if [ "${arg}" != "" ]; then
        if [ "${value}" = "" ]; then
        	command="${command} --${arg}"
        else
	        command="${command} --${arg} ${value}"
        fi
  fi

done < "$config_path"

run
