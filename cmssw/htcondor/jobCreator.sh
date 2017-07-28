

condorJob="condor${3%.sh}"
rm $condorJob
cat>> ${condorJob} <<condorjob

#Thisisacommentinthesubmitfile
#filename:doubletcondor
#universedescribesanexecutionenvirorment.Setuniversetovanilla universe=vanilla 

executable=$PWD/$3

condorjob

for ((i=$((1 + $4));i<=$(($2 + $4));i+=1))
        do
	end=$(($i))
	outputLogs="$PWD/outs/ml_dat_$i_$end.out"
	errorLogs="$PWD/errs/ml_dat_$i_$end.err"
	logLogs="$PWD/logs/ml_dat_$i_$end.log"
	
cat>> ${condorJob} <<condorjob2
	
arguments = $1 $i  


output=$outputLogs
error=$errorLogs
log=$logLogs
request_cpus=1
rank=Memory
queue
condorjob2
	

	done
	


