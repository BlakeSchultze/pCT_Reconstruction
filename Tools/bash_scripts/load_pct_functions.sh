#!/bin/bash
console_width=90
dark="0;"
light="1;"
red="31m"
green="32m"
brown="33m"
blue="34m"
purple="35m"
cyan="36m"
NoColor="\033[0m"
Black="\033[0;30m"
DarkGray="\033[1;30m"

Red="\033[0;31m"
LightRed="\033[1;31m"
Green="\033[0;32m"
LightGreen="\033[1;32m"
Brown="\033[1;33m"
Yellow="\033[0;33m"
Blue="\033[0;34m"
LightBlue="\033[1;34m"
Purple="\033[0;35m"
LightPurple="\033[1;35m"
Cyan="\033[0;36m"
LightCyan="\033[1;36m"
LightGray="\033[0;37m"
White="\033[1;37m"
separator_char="-"
default_background_color="6,40"
default_text_color="0,37"
reset_color="\033[0;37;6;40m"
###################################################################################################
########### User names of collaborators, including internal Baylor student user names #############
###################################################################################################
master="master"
blake="schultze"
blake_baylor="schultzeb"
paniz="karbasip"
ritchie="cair"
andriy="zatserkl"
pier="piersimonip"
tia="plautzt"
val="giamcomettiv"
vlad="bashkirovv"
current_user=$(id -un)
user_folder="/${current_user}"
recon_group="ionrecon"
###################################################################################################
######################## Paths/directories common to Kodiak/Tardis nodes ##########################
###################################################################################################
user_home=~
global_pct="/ion"
global_pct_homes="${global_pct}/home"
tardis_pct="/local"
workstation_pct="/home/share"
pct_code_folder="/pCT_code"
pct_data_folder="/pCT_data"
pct_docs_folder="/pCT_Documentation"
raw_data_folder="/raw_data"
pre_data_folder="/preprocessed_data"
proj_data_folder="/projection_data"
org_data_folder="/organized_data"
recon_data_folder="/reconstruction_data"
baylor_recon_repo="/pCT_Reconstruction"
rcode_folder="/Reconstruction"
pcode_folder="/Preprocessing"
incoming_folder="/incoming"
staging_folder="/staging"

git_code_folder="/git"
pct_collab_git_account="pCT-collaboration"
Baylor_git_account="BaylorICTHUS"
my_git_account="BlakeSchultze"

rcode_git_repo="pCT_Reconstruction"
old_rcode_git_repo="pct-recon-copy"

git_clone_addr_base="git@github.com:"
git_clone_addr_suffix=".git"

old_rcode_git_clone_addr="${git_clone_addr_base}${pct_collab_git_account}/${old_rcode_git_repo}${git_clone_addr_suffix}"
rcode_git_clone_addr="${git_clone_addr_base}${Baylor_git_account}/${rcode_git_repo}${git_clone_addr_suffix}"
my_rcode_git_clone_addr="${git_clone_addr_base}${my_git_account}/${rcode_git_repo}${git_clone_addr_suffix}"

#old_rcode_git_clone_addr="git@github.com:pCT-collaboration/pct-recon-copy.git"
#rcode_git_clone_addr="git@github.com:BaylorICThUS/pCT_Reconstruction.git"
#my_rcode_git_clone_addr="git@github.com:BlakeSchultze/pCT_Reconstruction.git"
current_rcode="/"
current_rdata="/ion/home/recon/pCT_data/reconstruction_data/CTP404_Sensitom/Simulated/G_15-05-24/0001/Output/15-05-24"
current_lrdata="/local/pCT_data/reconstruction_data/CTP404_Sensitom/Simulated/G_15-05-24/0001/Output/15-05-24"
#-------------------------------------------------------------------------------------------------#
#-std=c++11 -gencode arch=$rcode_compute,code=$rcode_sm
masterpcode=${pct_code_folder}${pcode_folder}/${master}
masterrcode=${pct_code_folder}${rcode_folder}/${master}
pcode_path=${pct_code_folder}${pcode_folder}
rcode_path=${pct_code_folder}${rcode_folder}
global_pcode_path=${global_pct}${pcode_path}
global_rcode_path=${global_pct}${rcode_path}
tardis_rcode_path=${tardis_pct}${rcode_path}
global_data_path=${global_pct}${pct_data_folder}
tardis_data_path=${tardis_pct}${pct_data_folder}


###################################################################################################
################### Tardis head/compute node IDs, nicknames, and IP addresses #####################
###################################################################################################
# IDs
tardis_head_node_ID="ecsn001"
tardis_compute_node1_ID="ecsn002"
tardis_compute_node2_ID="ecsn003"
tardis_compute_node3_ID="ecsn004"
tardis_compute_node4_ID="ecsn005"
# aliases
tardis_head_node_alias="whartnell"
tardis_compute_node1_alias="ptroughton"
tardis_compute_node2_alias="jpertwee"
tardis_compute_node3_alias="tbaker"
tardis_compute_node4_alias="pdavison"
# IPs
tardis_head_node_IP="192.168.225.1"
tardis_compute_node1_IP="192.168.225.2"
tardis_compute_node2_IP="192.168.225.3"
tardis_compute_node3_IP="192.168.225.4"
tardis_compute_node4_IP="192.168.225.5"
# Node ID numbers
kodiak=130
whartnell=0
ptroughton=1
jpertwee=2
tbaker=3
pdavison=4
###################################################################################################
########## Node specific CUDA SDK selection used in loading approprate gcc/nvcc modules ###########
###################################################################################################
version_CUDA_Kodiak="none"
version_CUDA_Tardis_Headnode=55
version_CUDA_Tardis=70
version_CUDA_workstation=70
rcode_cu_file="/src/pCT_Reconstruction_Data_Segments_Blake.cu"
rcode_compute="compute_35"
rcode_sm="sm_35"
rcode_object="recon.out"
rcode_flags="-O3"
alias nvcc11="nvcc -std=c++11 -gencode arch=${rcode_compute},code=${rcode_sm} ${rcode_flags}"
###################################################################################################
###################################### Function definitions #######################################
###################################################################################################
kodiak_modules=("purge" "unload gcc" "load gcc/4.8.4" "load geant4/10.1.1" "load openmpi/1.8.1" "load root" )
tardis_modules_old=("purge" "load cmake/3.2" "load java/1.8.0" "load emacs/24.5" "load geant4/10.1.1" "load git/2.4.3" "load make/4.1" "load matlab/2015a" "load perl/5.20" "load python/3.4" "load root" "load ruby/2.2.2" "load blacs/openmpi/gcc/64/1.1patch03" "load blacs/openmpi/open64/64/1.1patch03" "load blas/gcc/64/3.5.0" "load blas/open64/64/3.5.0" "load fftw3/openmpi/gcc/64/3.3.4" "load fftw3/openmpi/open64/64/3.3.4" "load lapack/gcc/64/3.5.0" "load lapack/open64/64/3.5.0" "load mpich/ge/gcc/64/3.1.4" "load mpich/ge/open64/64/3.1.4" "load mpiexec/0.84_432" "load mvapich2/1.9-gcc-4.9.2" "load mvapich2/gcc/64/2.1" "load mvapich2/open64/64/2.1" "load openblas/dynamic/0.2.14" "load openmpi/gcc/64/1.8.5" "load openmpi/open64/64/1.8.5" "load scalapack/gcc/64/1.8.0" "load scalapack/open64/64/1.8.0")
tardis_modules=("purge" "load cmake/3.2" "load java/1.8.0" "load emacs/24.5" "load geant4/10.1.1" "load git/2.4.3" "load make/4.1" "load matlab/2015a" "load perl/5.20" "load python/3.4" "load root" "load ruby/2.2.2" "load blacs/openmpi/gcc/64/1.1patch03" "load blacs/openmpi/open64/64/1.1patch03" "load blas/gcc/64/3.6.0" "load blas/open64/64/3.6.0" "load fftw3/openmpi/gcc/64/3.3.4" "load fftw3/openmpi/open64/64/3.3.4" "load lapack/gcc/64/3.6.0" "load lapack/open64/64/3.6.0" "load mpich/ge/gcc/64/3.2" "load mpich/ge/open64/64/3.2" "load mpiexec/0.84_432" "load mvapich2/1.9-gcc-4.9.2" "load mvapich2/gcc/64/2.2b" "load mvapich2/open64/64/2.2b" "load openblas/dynamic/0.2.15" "load openmpi/gcc/64/1.10.1" "load openmpi/open64/64/1.10.1" "load scalapack/mvapich2/gcc/64/2.0.2" "load scalapack/openmpi/gcc/64/2.0.2")
#    modules=("purge" "load cmake/3.2" "load java/1.8.0" "load emacs/24.5" "load geant4/10.1.1" "load git/2.4.3" "load make/4.1" "load matlab/2015a" "load perl/5.20" "load python/3.4" "load root" "load ruby/2.2.2" "load blacs/openmpi/gcc/64/1.1patch03" "load blacs/openmpi/open64/64/1.1patch03" "load blas/gcc/64/3.5.0" "load blas/open64/64/3.5.0" "load fftw3/openmpi/gcc/64/3.3.4" "load fftw3/openmpi/open64/64/3.3.4" "load lapack/gcc/64/3.5.0" "load lapack/open64/64/3.5.0" "load openblas/dynamic/0.2.14" "load openmpi/gcc/64/1.8.5" "load openmpi/open64/64/1.8.5" "load scalapack/gcc/64/1.8.0" "load scalapack/open64/64/1.8.0")
#    "load mvapich2/1.9"
#    "load mvapich2/1.9-centos"
#    "load mvapich2/1.9-gcc-4.7.2"
#    "load mvapich2/1.9-gcc-4.9.2"
#    "load mvapich2/1.9-intel"
#    "load mvapich2/1.9-intel-noaffinity"
#    "load mvapich2/2.0"
#    "load mvapich2/gcc/64/2.1"
#    "load mvapich2/open64/64/2.1"
###################################################################################################
###################################### Function definitions #######################################
###################################################################################################
function exe() { echo -e " $@" ; "$@" ; }
#function now() { date +"%T %Z %a %m-%d-%Y"; }
function dec2int() { echo "scale=0; $@/1" | bc ; }
function charrep() { printf '%*s' "$2" | tr ' ' "$1" ; }
function is_integer()
{
    re='^[0-9]+$'
    if [[ $1 =~ $re ]]; then return 0
    else return 1; fi
}
function current_node_name()
{
    node_ID=$HOSTNAME
    if [ $node_ID == "n130" ]; then node_alias="Kodiak"
    elif [ $node_ID == "ecsn001" -o $node_ID == "whartnell" ]; then node_alias="WHartnell"
    elif [ $node_ID == "ecsn002" ]; then node_alias="PTroughton"
    elif [ $node_ID == "ecsn003" ]; then node_alias="JPertwee"
    elif [ $node_ID == "ecsn004" ]; then node_alias="TBaker"
    elif [ $node_ID == "ecsn005" ]; then node_alias="PDavison"
    elif [ $node_ID == "tardis-student1" ]; then node_alias="Workstation #1"
    elif [ $node_ID == "tardis-student2" ]; then node_alias="Workstation #2"
    fi
    echo -e $node_alias
}
function node_info()
{
    local OPTIND
    node_name_flag='false'
    node_alias_flag='false'
    node_IP_flag='false'
    optstring='ain'
    while getopts $optstring opt; do
        case $opt in
            a) node_alias_flag='true';;
            i) node_IP_flag='true';;
            n) node_name_flag='true';;
            *) error "Unexpected option ${flag}";;
        esac
    done   
    # Set prefix of the variables corresponding to the specified node #: 0(head), 1-4 (compute nodes)
    node_num="${!#}"
    is_integer ${node_num} 
    if [ $? -eq 1 ]; then print "ERROR: Invalid node alias provided" 0,31 5,33; return 1; fi
    if test ${node_num} -eq 0; then target_base="tardis_head_node_"
    elif test ${node_num} -eq 1; then target_base="tardis_compute_node1_"
    elif test ${node_num} -eq 2; then target_base="tardis_compute_node2_"
    elif test ${node_num} -eq 3; then target_base="tardis_compute_node3_"
    elif test ${node_num} -eq 4; then target_base="tardis_compute_node4_"
    else
         print "ERROR: Invalid node ID # provided" 0,31 5,33       
         return 1
    fi   
    # Append suffix to variable name prefix corresponding to the node identifier desired: node's name, alias, or IP
    target_identifier="${target_base}ID"
    if ( "$node_IP_flag" ); then target_identifier="${target_base}IP"; fi
    if ( "$node_alias_flag" ); then target_identifier="${target_base}alias"; fi
    if ( "$node_name_flag" ); then target_identifier="${target_base}ID"; fi
    REPLY=${!target_identifier} 
}
function set_color()
{
    if [ -z "$1" ]; then
        text_color_shade_code="0";
        text_color_color_code="32";
    else
        text_color_code_length=${#1};
        text_color_shade_code=${1:0:1};
        text_color_color_code=${1:2};
    fi;
    if [ -z "$2" ]; then
        background_shade_code="6";
        background_color_code="40";
    else
        background_color_code_length=${#2};
        background_color_shade_code=${2:0:1};
        background_color_color_code=${2:2};
    fi;
    if [ -z "$3" ]; then
        underline_code="";
    else
       underline_code="\033[4m";
    fi;
    #echo -e "\033[${text_color_shade_code};${text_color_color_code};${background_color_shade_code};${background_color_color_code}m${underline_code}"
    REPLY="\033[${text_color_shade_code};${text_color_color_code};${background_color_shade_code};${background_color_color_code}m${underline_code}"
}
function color_text()
{
  if [ -z "$2" ]; then
      set_color ${default_text_color} ${default_background_color}
  else
      if [ -z "$3" ]; then
          set_color $2 ${default_background_color}
      else
          set_color $2 $3
      fi
  fi
  if [ -z "$4" ]; then
        underline_code="";
    else
       underline_code="\033[4m";
    fi;
  color_cmd=${REPLY}
  REPLY="${color_cmd}${underline_code}${1}\e[m"
  #echo -e $REPLY
}
function print()
{
  if [ -z "$2" ]; then
      set_color ${default_text_color} ${default_background_color}
  else
      if [ -z "$3" ]; then
          set_color $2 ${default_background_color}
      else
          set_color $2 $3
      fi
  fi
  if [ -z "$4" ]; then
        underline_code="";
    else
       underline_code="\033[4m";
    fi;
  echo -e "${REPLY}${underline_code}${1}\e[m"
}
function print_alias()
{
    equal_pos=$(expr index "$2" = )
    var_length=$(expr $equal_pos - 1)
    cmd="${1}"
    var=${2:0:$var_length}
    val=${2:${equal_pos}}
    alias_val="${var}=${val} $3"
    echo -e "${LightRed}-> ${Brown}alias ${Green}${var}=${LightPurple}${val} $3${NoColor}"
    alias "${alias_val}"
}
function print_info() { echo -e "${Green}$1: ${LightCyan}$2 " ; }
function print_label() { echo -e "${Green}$@: ${LightCyan}" ; }
function print_module_load()
{
    words=( $1 )
    module_command=${words[0]}
    module_name=${words[1]}
    echo -e "${LightRed}-> ${Brown}module ${LightBlue}$module_command ${LightPurple}$module_name${LightCyan}"
    module $1 $2
}
function print_newline() { echo -e ""; }
function print_result() { echo -e "${LightCyan}$@ ${NoColor}" ; }
function print_section_header()
{
    heading_length=${#1}
    num_dashes=$(expr $console_width - $heading_length - 2)
    num_front=$(expr $num_dashes / 2)
    num_back=$(expr $num_dashes - $num_front)
    front_string=$(charrep $separator_char $num_front)
    back_string=$(charrep $separator_char $num_back)
    separator_string=$(charrep "-" $console_width)
    if [ -z "$2" ]; then
       print "${separator_string}" ${default_text_color} ${default_background_color}
       print "${front_string} $1 ${back_string}" ${default_text_color} ${default_background_color}
       print "${separator_string}" ${default_text_color} ${default_background_color}   
    else
        if [ -z "$3" ]; then
            print "${separator_string}" ${2} ${default_background_color}
            print "${front_string} $1 ${back_string}" ${2} ${default_background_color}
            print "${separator_string}" ${2} ${default_background_color}   
        else
            print "${separator_string}" ${2} ${3}
            print "${front_string} $1 ${back_string}" ${2} ${3}
            print "${separator_string}" ${2} ${3}   
        fi
    fi
    print_newline
}
function print_section_title()
{
    heading_length=${#1}
    num_dashes=$(expr $console_width - $heading_length - 2)
    num_front=$(expr $num_dashes / 2)
    num_back=$(expr $num_dashes - $num_front)
    front_string=$(charrep $separator_char $num_front)
    back_string=$(charrep $separator_char $num_back)
    separator_string=$(charrep "-" $console_width)
    if [ -z "$2" ]                           # If no 2nd parameter present
    then
        heading_separator="$Brown$separator_string"
        echo -e $heading_separator
        echo -e "$Brown$front_string $1 $back_string"
        echo -e $heading_separator
    else
        heading_separator="$2$separator_string"
        echo -e $heading_separator
        echo -e "${2}$front_string $1 $back_string"
        echo -e $heading_separator
  fi
}
function print_program_version()
{
    echo -e "${Green}$1 version:${LightCyan}"
    echo -e "$( $2 )${NoColor} $3"
}
function scp_recon()
{   
    echo -e ${REPLY}
    scp ${user_home}${rcode_path}${user_folder}/src ${1}:${tardis_pct}${rcode_path}${user_folder}
    scp ${user_home}${rcode_path}${user_folder}/include ${1}:${tardis_pct}${rcode_path}${user_folder}
    print_newline
}
function txrecon2all()
{
    print_section_header "Transferring reconstruction code to all Tardis compute nodes" 1,33  5,40
    print "\nCurrent user: " 0,32;  print "${current_user}" 0,35
    print "\nTransferring to: " 0,32;  print "${tardis_compute_node1_alias}" 0,35
    set_color ${default_text_color} ${default_background_color} 
    echo -e ${REPLY}
    scp ${user_home}${rcode_path}${user_folder}/src ${current_user}@${tardis_compute_node1_alias}:${tardis_pct}${rcode_path}${user_folder}
    scp ${user_home}${rcode_path}${user_folder}/include ${current_user}@${tardis_compute_node1_alias}:${tardis_pct}${rcode_path}${user_folder}
    print "\nTransferring to: " 0,32;  print "${tardis_compute_node2_alias}" 0,35
    set_color ${default_text_color} ${default_background_color}
    echo -e $REPLY
    scp ${user_home}${rcode_path}${user_folder}/src ${current_user}@${tardis_compute_node2_alias}:${tardis_pct}${rcode_path}${user_folder}
    scp ${user_home}${rcode_path}${user_folder}/include ${current_user}@${tardis_compute_node2_alias}:${tardis_pct}${rcode_path}${user_folder}
    print "\nTransferring to: " 0,32;  print "${tardis_compute_node3_alias}" 0,35
    set_color ${default_text_color} ${default_background_color}
    echo -e $REPLY
    scp ${user_home}${rcode_path}${user_folder}/src ${current_user}@${tardis_compute_node3_alias}:${tardis_pct}${rcode_path}${user_folder}
    scp ${user_home}${rcode_path}${user_folder}/include ${current_user}@${tardis_compute_node3_alias}:${tardis_pct}${rcode_path}${user_folder}
    print "\nTransferring to: " 0,32;  print "${tardis_compute_node4_alias}" 0,35
   set_color ${default_text_color} ${default_background_color}
    echo -e $REPLY
    scp ${user_home}${rcode_path}${user_folder}/src ${current_user}@${tardis_compute_node4_alias}:${tardis_pct}${rcode_path}${user_folder}
    scp ${user_home}${rcode_path}${user_folder}/include ${current_user}@${tardis_compute_node4_alias}:${tardis_pct}${rcode_path}${user_folder}
}
function txrecon()
{   # Process the execution parameters/arguments
    local OPTIND
    node_num_flag='true'
    while getopts 'a:' opt; do
        case $opt in
             a) node_num_flag='false'
             node_num=${!OPTARG};;            
            *) error "Unexpected option ${flag}";;
        esac
    done   
    
    # Set node # to last execution argument, convert this to  name and verify a valid node # was provided
    if ( "$node_num_flag" ); then node_num=${!#}; fi
    node_info -n $node_num;
    if [ $? -eq 1 ]; then return; fi
    node_name=$REPLY 
    node_info -a $node_num; node_alias=$REPLY
    node_info -i $node_num; node_IP=$REPLY 
    
    # Print destination node and initiate reconstruction code transfer
    color_text "${node_name}/${node_alias} (${node_IP})" 0,33 6,40 
    target_node_statement=" ${REPLY}"
    color_text "\nTransferring reconstruction code to:" 0,32 5,40 4
    echo -e ${REPLY}${target_node_statement}
    set_color 0,35 5,40 
    scp_recon ${node_IP}  
}
function lsm()
{
    local OPTIND
    match_string='*'
    depth='1'
    filename_string=''
    filename='lsm_results.txt'
    verbose_string=''
    mtime='' 
    mtime_string=''
    file_type='d'
    period_identifier="days"
    age=''
    while getopts 'vm:d:f:o:t:sIMCAR' opt; do
        case $opt in
            v) verbose_string='-printf %f\n';;
            m) match_string=${OPTARG};;
            d) depth=${OPTARG};;
            f) filename=${OPTARG};;
            o) mtime=${OPTARG};;
            t) file_type=${OPTARG};;
            I) filename='ROI_analysis_folders.txt';;
            M) filename='multiplot_folders.txt';;
            C) filename='compared_folders.txt';;
            A) filename='averaging_folders.txt';;
            R) filename='reconstruction_folders.txt';;
            s) filename_string="$(echo -e "-fprintf ${filename} ")%f\n";;
            *) error "Unexpected option ${flag}";;
        esac
    done
    modified_qualifier=${mtime:0:1}   
    if [[ $modified_qualifier == "-" ]]; then age_identifier="less than"; age=${mtime:1}
    elif [[ $modified_qualifier == "+" ]]; then age_identifier="greater than"; age=${mtime:1}
    else age_identifier="exactly"; age=${mtime:0}; fi
    if [[ ${age} == 1 ]]; then period_identifier="day"; fi
    color_text " $match_string" 0,33 6,40; matching=$REPLY
    color_text " $depth"        0,33 6,40; maxdepth=$REPLY
    color_text " $filename"     0,33 6,40; filenaming=$REPLY
    color_text " ${age_identifier} ${age} ${period_identifier} ago"     0,33 6,40; ordering=$REPLY
    print "Searching for names matching:$matching" 0,32 6,40 underline
    print "Maximum search depth:$maxdepth"         0,32 6,40 underline
    if [[ -n $mtime ]]; then print "Only those created/modified:${ordering}" 0,32 6,40 underline; mtime_string="$(echo -e "-mtime $mtime")"; fi
    if [[ -n $filename_string ]]; then print "Saving results to:$filenaming" 0,32 6,40 underline; filename_string="$(echo -e "-fprintf ${filename} ")%f\n"; fi
    num_results=$(find -maxdepth $depth -type $file_type -name "${match_string}" ${mtime_string} ${verbose_string} ${filename_string} | wc -l)
    search_results=$(find -maxdepth $depth -type $file_type -name "${match_string}" ${mtime_string} ${verbose_string} ${filename_string})
    color_text " $num_results"     0,33 6,40; num_matching=$REPLY
    color_text "${search_results}" 0,35 6,40; results=$REPLY
    print "Search matches returned:$num_matching" 0,32 6,40 underline
    print "Search results:\n$results" 0,32 6,40 underline
    print_newline
}
function add_rcode_repo()
{
    local OPTIND
    username="$(id -un)"
    account=${pct_collab_git_account}
    repo=${old_rcode_git_repo}
    while getopts 'u:a:r:' opt; do
        case $opt in
            u) username=${OPTARG};;
            a) account=${OPTARG};;
            r) repo=${OPTARG};;           
            *) error "Unexpected option ${flag}";;
        esac
    done
    rcode_path="${tardis_rcode_path}/${username}${git_code_folder}/${account}/"
    mkdir -p ${rcode_path}
    cd ${rcode_path}
    
    rcode_git_clone_addr="${git_clone_addr_base}${account}/${repo}${git_clone_addr_suffix}"
    git clone ${rcode_git_clone_addr}   
}
function create_recon_user()
{
    local OPTIND
    username="$(id -un)"
    while getopts 'u:a:r:' opt; do
        case $opt in
            u) username=${OPTARG};;
            *) error "Unexpected option ${flag}";;
        esac
    done    
    old_rcode_path="${tardis_rcode_path}/${username}${git_code_folder}/${pct_collab_git_account}/"
    mkdir -p ${old_rcode_path}
    cd ${old_rcode_path}
    git clone ${old_rcode_git_clone_addr}   
}
function nvccgen()
{
    #nvcc11="nvcc -std=c++11 -gencode arch=compute_35,code=sm_35 -O3"
    print_newline
    color_prefix="\033[0;34;5;47m"
    color_postfix="\033[3;40;5;1;34m"
    target_source_code="${tardis_pct}${rcode_path}${user_folder}${rcode_cu_file}"
    if [ -z "$1" ] # If no 2nd parameter present
    then
        print_section_header "Using compute/sm 35 (default)" 0,34 5,47
        nvcc_cmd="nvcc -std=c++11 -gencode arch=$rcode_compute,code=$rcode_sm ${rcode_flags} ${target_source_code} -o ${rcode_object}"
        print "${nvcc_cmd}" 1,33 5,40
        set_color 0,31 5,40; echo -e $REPLY
        echo -e "$(${nvcc_cmd})"
    else
        if [ -z "$2" ]
        then
            print_section_header "Using compute/sm ${1} (default)" 0,34 5,47
            nvcc_cmd="nvcc -std=c++11 -gencode arch=compute_$1,code=sm_$1 ${rcode_flags} ${target_source_code} -o ${rcode_object}"
            print "${nvcc_cmd}" 1,33 5,40
            set_color 0,31 5,40; echo -e $REPLY
            echo -e "$(${nvcc_cmd})"      
        else
            if [ $1 -gt $2 ]
            then
                heading_separator="\033[${Red}$separator_string\033[0m"
                echo -e $heading_separator
                echo -e "${Red}ERROR:${LightRed} Incompatible code generation requested\n"
                echo -e "${LightRed}\tThe selected target SM architecture (sm_$2) only supports code generation from compute_$2 or an older instruction set"
                echo -e $heading_separator
            else
                print_section_header "Using compute_${1}/sm_${2}" 0,34 5,47

                nvcc_cmd="nvcc -std=c++11 -gencode arch=compute_$1,code=sm_$2 ${rcode_flags} ${target_source_code} -o ${rcode_object}"
                print "${nvcc_cmd}" 1,33 5,40
                set_color 0,31 5,40; echo -e $REPLY
                echo -e "$(${nvcc_cmd})"      
            fi
        fi
    fi
}
function runrecon()
{
      if [ -z "$1" ]
      then
         nvccgen
         ./recon.out
         else
         if [ -z "$2" ]
         then
             nvccgen $1 $1
             ./recon.out
         else
             if [ $1 -gt $2 ]
             then
                 echo -e "${Red}  ERROR:${LightRed} Incompatible code generation requested\n"
                 echo -e "${LightRed}\tTarget SM architecture (sm_$2) only supports code generation from compute_$2 or an older instruction set"
             else
                 nvccgen $1 $2
                 ./recon.out
             fi
         fi
     fi
}
function load_CUDA_modules()
{
    print "\nLoading modules required by selected version of CUDA..." 0,32
    #---------------------------------------------------------------------------------------------#
    if [ $HOSTNAME == "whartnell" ]
    then
        load_CUDA=$version_CUDA_Tardis_Headnode
    #---------------------------------------------------------------------------------------------#
    elif [ $HOSTNAME == "ecsn002" -o $HOSTNAME == "ecsn003" -o $HOSTNAME == "ecsn004" -o $HOSTNAME == "ecsn005" ]
    then
        load_CUDA=$version_CUDA_Tardis
    #---------------------------------------------------------------------------------------------#
    elif [ $HOSTNAME == "tardis-student1" -o $HOSTNAME == "tardis-student2" ]
    then
        load_CUDA=$version_CUDA_Workstation
    else
        load_CUDA="none"
    fi
    #---------------------------------------------------------------------------------------------#
    if [ ${load_CUDA} == 55 ]
    then
        print_module_load "unload gcc"
        print_module_load "load cuda55"
          print_module_load "load cuda55/blas/5.5.22"
          print_module_load "load cuda55/fft/5.5.22"
          print_module_load "load cuda55/toolkit/5.5.22"
    #---------------------------------------------------------------------------------------------#
    elif [ ${load_CUDA} == 65 ]
    then
        print_module_load "unload gcc"
        print_module_load "load gcc/4.8.4"
        print_module_load "load cuda65/blas/6.5.14"
        print_module_load "load cuda65/fft/6.5.14"
        print_module_load "load cuda65/gdk/340.29"
          print_module_load "load cuda65/nsight/6.5.14"
          print_module_load "load cuda65/profiler/6.5.14"
        print_module_load "load cuda65/toolkit/6.5.14"
    #---------------------------------------------------------------------------------------------#
    elif [ ${load_CUDA} == 70 ]
    then
        print_module_load "unload gcc"
        print_module_load "load gcc/4.9.2"
        print_module_load "load cuda70"
          print_module_load "load cuda70/blas/7.0.28"
          print_module_load "load cuda70/fft/7.0.28"
          print_module_load "load cuda70/gdk/346.46"
          print_module_load "load cuda70/nsight/7.0.28"
        print_module_load "load cuda70/profiler/7.0.28"
          print_module_load "load cuda70/toolkit/7.0.28"
    fi
}
function find_git_branch()
{
  # Based on: http://stackoverflow.com/a/13003854/170413
  local branch
  if branch=$(git rev-parse --abbrev-ref HEAD 2> /dev/null); then
    if [[ "$branch" == "HEAD" ]]; then
      branch='detached*'
    fi
    git_branch="($branch)"
  else
    git_branch=""
  fi
}

function find_git_dirty()
{
  local status=$(git status --porcelain 2> /dev/null)
  if [[ "$status" != "" ]]; then
    git_dirty='* '
  else
    git_dirty=''
  fi
}
function can_user_write_to_file()
{
    if [[ $# -lt 2 || ! -r $2 ]]
    then
        echo 0
        return
    fi
    local user_id=$(id -u $1 2>/dev/null)
    local file_owner_id=$(stat -c "%u" $2)
    if [[ ${user_id} == ${file_owner_id} ]]
    then
        echo 1
        return
    fi
    local file_access=$(stat -c "%a" $2)
    local file_group_access=${file_access:1:1}
    local file_group_name=$(stat -c "%G" $2)
    local user_group_list=$(groups $1 2>/dev/null)
    if [[ ${file_group_access} -ge 6 ]]
    then
        for el in ${user_group_list-nop}
        do
            if [[ "${el}" == ${file_group_name} ]]
            then
                echo 1
                return
            fi
        done
    fi
    echo 0
}
###################################################################################################
############################### Establish ssh connection shortcuts ################################
###################################################################################################
# ssh aliases
alias gokodiak="ssh ${current_user}@kodiak.baylor.edu"
alias gowhartnell="ssh ${current_user}@whartnell"
alias gojpertwee="ssh ${current_user}@jpertwee"
alias gotbaker="ssh ${current_user}@tbaker"
alias goptroughton="ssh ${current_user}@ptroughton"
alias gopdavison="ssh ${current_user}@pdavison"
alias gows1="ssh schultzeb@tardis-student1.ecs.baylor.edu"
alias gows2="ssh schultzeb@tardis-student2.ecs.baylor.edu"
current_node_alias=$(current_node_name )
###################################################################################################
########################### Define shortcut directory change commands #############################
###################################################################################################
alias gocode="cd ${global_pct}${pct_code_folder}"                              #
alias gopcode="cd ${global_pcode_path}"                                       #
alias gorcode="cd ${global_rcode_path}"                                        #
alias gomasterpcode="cd ${global_pct}${masterpcode}"                           #
alias gomasterrcode="cd ${global_pct}${masterrcode}"                          #
alias gogitpcode="cd ${global_pcode_path}${user_folder}"                       #
alias gogitrcode="cd ${global_rcode_path}${user_folder}${baylor_recon_repo}"   #
alias godata="cd ${global_data_path}"                                          #
alias goraw="cd ${global_data_path}${raw_data_folder}"                         #
alias gopre="cd ${global_data_path}${pre_data_folder}"                         #
alias goproj="cd ${global_data_path}${proj_data_folder}"                       #
alias gorecon="cd ${global_data_path}${recon_data_folder}"                     #
alias goorg="cd ${global_data_path}${org_data_folder}"                         #
alias godocs="cd ${global_data_path}${pct_docs_folder}"                       #
alias goinc="cd ${global_pct}${incoming_folder}"                               #
alias gostage="cd ${global_pct}${staging_folder}"                              #
alias gomyinc="cd ${global_pct}${incoming_folder}${user_folder}"               #
alias gomystage="cd ${global_pct}${staging_folder}${user_folder}"              #
alias gonewrecon="cd ${current_rdata}"                                         #

alias golpct="cd ${tardis_pct}"                                                #
alias golcode="cd ${tardis_pct}${pct_code_folder}"                             #
alias golrcode="cd ${tardis_pct}${rcode_path}${user_folder}"                   #
alias goldata="cd ${tardis_data_path}"                                         #
alias golorg="cd ${tardis_data_path}${org_data_folder}"                        #
alias golrecon="cd ${tardis_data_path}${recon_data_folder}"                    #
alias golnewrecon="cd ${current_lrdata}"                                         #

alias gomycode="cd ${user_home}${pct_code_folder}"                             #
alias gomypcode="cd ${user_home}${pcode_path}${user_folder}"                   #
alias gomyrcode="cd ${user_home}${rcode_path}${user_folder}"                   #
alias gomydata="cd ${user_home}${pct_data_folder}"                             #
alias gomyorg="cd ${user_home}${pct_data_folder}${org_data_folder}"            #
alias gomyrecon="cd ${user_home}${pct_data_folder}${recon_data_folder}"        #

alias gogrp="cd ${recon_group}"                                        #
