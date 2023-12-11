#!/bin/bash

networks="AsnetAm_0_1_1 giul39_0_1_1 Oxford_0_1_1 AttMpls_0_1_1 Iij_0_1_1 zib54_0_1_1 Chinanet_0_1_1 Ntt_0_1_1 india35_0_1_1"
model="model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1"
modelbs50="model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1"

for net in $networks; do
    echo "$net"
    mkdir -p models/"$net"
    rsync -q lipn-gpu:/data1/schulz/models/"$net"/"$model"/best_checkpoint.bson models/"$net"/
    echo "Return code : $?"
    if [[ $? -ne 0 ]]
    then
	    echo "Error"
            rsync -q lipn-gpu:/data1/schulz/models/"$net"/"$modelbs50"/best_checkpoint.bson models/"$net"/
    fi
    #echo "$net"_frod
    #mkdir -p models/"$net"_frod
    #rsync -q lipn-gpu:/data1/schulz/models/"$net"_frod/"$model"/best_checkpoint.bson models/"$net"_frod/
    #echo "$net"_delay
    #mkdir -p models/"$net"_delay
    #rsync -q lipn-gpu:/data1/schulz/models/"$net"_delay/"$model"/best_checkpoint.bson models/"$net"_delay/
    #echo "$net"_delay_frod
    #mkdir -p models/"$net"_delay_frod
    #rsync -q lipn-gpu:/data1/schulz/models/"$net"_delay_frod/"$model"/best_checkpoint.bson models/"$net"_delay_frod/
    sleep 5
done
