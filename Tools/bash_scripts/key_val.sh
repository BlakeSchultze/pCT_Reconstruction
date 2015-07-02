#! /bin/bash
inputs=("$@")
numInputs=$#

COUNTER=0
keywords=('bool' 'const' 'int' 'long' 'unsigned' 'float' 'double')
numKeywords=${#keywords[@]}
#echo $numKeywords


while [ $COUNTER -lt $# ]; do
if [ "${inputs[$COUNTER+3]}" == "n" ]
then
	fileName=${inputs[COUNTER]}
	oldValue=$(cat "$fileName" | grep -i ${inputs[$COUNTER+1]} | tr -d '[A-Za-z#=;:_></ ]')
	oldVal2="$oldValue"
	echo $oldVal2
	echo $oldValue
	newValue="${inputs[$COUNTER+2]}"
	sed -i "s/$oldValue/$newValue/g" "$fileName"

#elif [ "${inputs[$COUNTER+3]}" == "s" ]
#then
#	i=0
#	fileName=${inputs[COUNTER]}
	#line=$(sed -e "g/${inputs[$COUNTER+1]}/" "$fileName")
#	line=$(cat "$fileName" | grep -i ${inputs[$COUNTER+1]} | tr -d '[=;></0-9]' )
#	echo $line
	#tempLine=$(echo "$line" | tr -d '[=;></0-9 ]')	

#	while [ $i -lt $numKeywords ]; do
#		temp=$(echo "$tempLine" | sed -e "s/${keywords[$i]}//g" )
#		echo $temp
#		unset $tempLine
#		$tempLine=${tempLine:-$temp}
#		unset $temp
#		echo $tempLine
#		let i=i+1
#	done

	#oldValue="$tempLine"
	#echo $oldValue
	#newValue="${inputs[$COUNTER+2]}"
	#echo $newValue
	#sed -i "s/$oldValue/$newValue/g" "$fileName"
else
	echo "WRONG ARGUMENTS"
fi
let COUNTER=COUNTER+4
done

