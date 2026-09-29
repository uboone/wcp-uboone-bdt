// cz: code modified from tutorials/tmva/TMVAClassification.C

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <set>
#include <limits>
#include <unordered_map>
#include <algorithm>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "WCPLEEANA/Util.h"

#include "WCPLEEANA/eval.h"

using namespace std;
using namespace LEEana;

#include "WCPLEEANA/tree_wrangler.h"


struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};
using Key = std::pair<int,int>;
using Value = std::pair< bool, std::unordered_map<int,std::vector<int>> >;


void print_help() {
  std::cout << R"(

========================================
 tree_trimmer : Help
========================================

Overview:
---------
tree_trimmer reads a ROOT file containing multiple directories and TTrees and produces a
filtered output file. It organizes events by (run, subrun, event), removes
duplicates (optionally), applies data-quality and selection cuts, and ensures
POT consistency by operating at the subrun level.

The program:
  - Loads trees defined in a config file (run tree_trimmer -H for more info)
  - Optionally filters based on Wire-Cell BDT training lists
  - Removes bad runs and detector-quality failures
  - Writes a trimmed ROOT file with consistent POT accounting

Usage:
------
  tree_trimmer <input_file> <output_file> <config.txt> [options]

Required arguments:
-------------------
  input_file     Input ROOT file
  output_file    Output ROOT file
  config.txt     Tree configuration file

Options:
--------

  -h
      Show this help message and exit

  -H
      Show help message for configuration file and exit

  -d<char>
      Set delimiter used in config file (default: ',')

  -o<int>
      Overwrite output file if it exists
        0 = do not overwrite (default)
        1 = overwrite

  -k<int>
      Duplicate handling:
        0 = allow duplicates
        1 = remove duplicates (default)
        2 = exit if duplicates are found

  -m<int>
      Maximum number of events to save
      (rounded up to nearest subrun for correct POT)

  -e<int>
      Start saving from this event index
      (rounded down to nearest subrun)

  -y<int>
      Start saving events at this run number

  -z<int>
      Stop saving events at this run number

  -w<int>
      Start saving events at this subrun number (within the starting run)

  -x<int>
      Stop saving events at this subrun number (within the stopping run)

  -s<int>
      Skip data-quality cuts:
        0 = keep all runs (ignore good run list)
        1 = apply good run list cuts (default)

  -n<int>
      Beam selection:
        0 = BNB (default)
        1 = NuMI

  -c<int>
      Data type:
        0 = MC (default)
        1 = Data

  -r<int>
      Lantern failure handling:
        0 = keep subruns where Lantern failed
        1 = remove subruns where Lantern failed (default)

  -l<string>
      Path to Wire-Cell BDT training list file

  -i<string>
      Path to list of runs which will additionally be removed
      Format: <run> <subrun1> <subrun2> ...
      Use <run> -1 to remove all subruns in the run

  -g<string>
      Global file type label used with training list

  -b<int>
      Filter based on BDT training usage:
        -1 = keep all subruns (default)
         0 = keep only subruns NOT used in training
         1 = keep only subruns used in training

      NOTE: Requires both -l and -g

  -a<string>
      Set SAM definition string to be stored in output trees

  -v<int>
      Verbosity level for event loop (default: 10000)

  -u<int>
      Verbosity level for POT loop (default: 1000)

Configuration File:
-------------------
run tree_trimmer -H for more info

Examples:
---------

  Basic usage:
    tree_trimmer input.root output.root config.txt

  Overwrite output and limit to 10000 events:
    tree_trimmer input.root output.root config.txt -o1 -m10000

  Keep only BDT training subruns:
    tree_trimmer input.root output.root config.txt -ltrain.txt -gmytype -b1

  Run on data with quality cuts:
    tree_trimmer input.root output.root config.txt -c1 -s1

Notes:
------
- Event selection is applied at the subrun level to preserve POT consistency.
- Duplicate events are identified via (run, subrun, event).
- Some run ranges are hard-coded as bad and will always be removed unless -s0.

)";
}




bool keep_subrun(int run, int subrun,
                 int start_run, int start_subrun, int stop_run, int stop_subrun,
                 int remove_lantern_fails, bool haveReco, 
                 int flag_keep_only_bdt_train, const std::string& global_file_type, const std::map<string, std::set<std::pair<int, int>>>& map_type_run_subrun, 
                 std::unordered_map<int,std::vector<int>>& remove_individual_run, 
                 int skip_cut, int flag_data, int flag_numi,
                 const std::set<int>& good_runlist_set, const std::set<int>& low_lifetime_set, const std::set<int>& low_neutrino_count_numi_run2RHC_set) {

  bool flag_keep_subrun = false;

  if ( ( run<start_run ) || ( run==start_run && subrun<start_subrun) ) {
      return flag_keep_subrun;
  } 
  if ( ( run>stop_run  ) || ( run==stop_run  && subrun>stop_subrun ) ) {
      return flag_keep_subrun;
  } 

  // check if the lantern container failed on this subrun
  if (remove_lantern_fails == 1 && haveReco == 0) {
      return flag_keep_subrun;
  }

  // Cutting or including WireCell BDT training subruns
  if (flag_keep_only_bdt_train >= 0) {
      auto filetype_it = map_type_run_subrun.find(global_file_type);
      if (filetype_it != map_type_run_subrun.end()) {
          auto rs_it = filetype_it->second.find(std::make_pair(run, subrun));
          if (rs_it != filetype_it->second.end() && flag_keep_only_bdt_train == 0) {
              return flag_keep_subrun;
          }
          if (rs_it == filetype_it->second.end() && flag_keep_only_bdt_train == 1) {
              return flag_keep_subrun;
          }
      } else if (flag_keep_only_bdt_train == 1) {
          return flag_keep_subrun;
      }
  }

  // Remove runs if they are in the extra list provided 
  auto rs_it = remove_individual_run.find(run);
  if (rs_it != remove_individual_run.end()) {
    // Removing all subruns in this run
    if ((*rs_it).second.at(0)==-1){
      return flag_keep_subrun;
    }
    // Check if this subrun is in the list of ones to remove in the given run 
    auto s_it = std::find((*rs_it).second.begin(), (*rs_it).second.end(), subrun);
    if (s_it != (*rs_it).second.end()) {
      return flag_keep_subrun;
    }
  }

  // Remove bad run-subruns if the flag is set.
  if (flag_data == 1 && skip_cut == 0) {
      if (good_runlist_set.find(run) == good_runlist_set.end()) {
          return flag_keep_subrun;
      }
      if (low_lifetime_set.find(run) != low_lifetime_set.end()) {
          return flag_keep_subrun;
      }
      if (flag_numi == 1 && low_neutrino_count_numi_run2RHC_set.find(run) != low_neutrino_count_numi_run2RHC_set.end()) {
          return flag_keep_subrun;
      }
      if (run >= 7004 && run <= 7070) {
          return flag_keep_subrun;
      }
      if (run >= 8321 && run <= 8404) {
          return flag_keep_subrun;
      }
      if (run >= 15369 && run <= 15402) {
          return flag_keep_subrun;
      }
  }

  flag_keep_subrun = true;
  return flag_keep_subrun;

}


int main( int argc, char** argv )
{
  if(argc==2 && argv[1][1]=='h'){
    print_help();
    return 0;
  }
  else if(argc==2 && argv[1][1]=='H'){
    print_help_wrangler_config(true);
    return 0;
  }
  else if (argc < 4) {
    std::cout << "tree_trimmer #input_file #output_file #config.txt" << std::endl;
    std::cout << "tree_trimmer -h for further help and instructions." << std::endl;
    return -1;
  }


  TString input_file = argv[1];
  TString out_file = argv[2];

  std::string config_file_name=argv[3];

  char delimiter = ',';

  int flag_overwrite=0;

  int flag_kill_duplicates=1;

  int max_events=std::numeric_limits<int>::max();
  int start_events=0;
  int start_run=0;
  int start_subrun=0;
  int stop_run=std::numeric_limits<int>::max();
  int stop_subrun=std::numeric_limits<int>::max();

  int skip_cut = 1;
  int flag_numi = 0;
  int flag_data = 0;

  int remove_lantern_fails = 1;

  TString training_list = "";
  string global_file_type = "";
  int flag_keep_only_bdt_train = -1;

  TString remove_individual_run_list = "";

  int flag_set_samdef = 0;
  TString samdef="";

  int set_verbose=10000;
  int set_verbose_pot=1000;

  for (Int_t i = 4; i < argc; ++i) {

    // Skip non-flags
    if (argv[i][0] != '-') continue;

    char flag = argv[i][1];
    char* value_ptr = nullptr;

    // Case 1: attached value (-o1)
    if (argv[i][2] != '\0') {
      value_ptr = &argv[i][2];
    }
    // Case 2: separate value (-o 1)
    else if (i + 1 < argc && argv[i+1][0] != '-') {
      value_ptr = argv[i + 1];
      ++i; // consume next argument
    }

    // Guard against missing values
    if (!value_ptr && flag != 'a') {
      std::cerr << "Missing value for -" << flag << std::endl;
      continue;
    }

    switch(flag){

    case 'd':
      delimiter = value_ptr ? value_ptr[0] : delimiter;
      break;

    case 'o':
      if (value_ptr) flag_overwrite = atoi(value_ptr);
      break;

    case 'k':
      if (value_ptr) {
        flag_kill_duplicates = atoi(value_ptr);
        if(flag_kill_duplicates==0) { std::cout<<"Allowing duplicates if found.\n\n"; }
        else if(flag_kill_duplicates==1) { std::cout<<"Will remove duplicates if found.\n\n"; }
        else if(flag_kill_duplicates==2) { std::cout<<"Will exit if duplicates are found.\n\n"; }
        else {
          std::cout<<"Unknown -k option, setting to default flag_kill_duplicates=1\n\n";
          flag_kill_duplicates=1;
        }
      }
      break;

    case 'm':
      if (value_ptr) {
        max_events = atoi(value_ptr);
        if(max_events < 0) max_events = 0;
        std::cout<<"Will be saving at most "<<max_events<<" events to the output file.\n";
        std::cout<<"Note that this will ''round up'' to the nearest subrun in order to ensure the POT is correct.\n\n";
      }
      break;

    case 'e':
      if (value_ptr) {
        start_events = atoi(value_ptr);
        std::cout<<"Will start saving to the output file at event "<<start_events<<"\n";
        std::cout<<"Note that this will ''round down'' to the nearest subrun in order to ensure the POT is correct.\n\n";
      }
      break;

    case 'w':
      if (value_ptr) {
        start_subrun = atoi(value_ptr);
        std::cout<<"Will start saving to the output file at subrun "<<start_subrun<<"\n\n";
      }
      break;

    case 'x':
      if (value_ptr) {
        stop_subrun = atoi(value_ptr);
        std::cout<<"Will stop saving to the output file at subrun "<<stop_subrun<<"\n\n";
      }
      break;

    case 'y':
      if (value_ptr) {
        start_run = atoi(value_ptr);
        std::cout<<"Will start saving to the output file at run "<<start_run<<"\n\n";
      }
      break;

    case 'z':
      if (value_ptr) {
        stop_run = atoi(value_ptr);
        std::cout<<"Will stop saving to the output file at run "<<stop_run<<"\n\n";
      }
      break;

    case 's':
      if (value_ptr) {
        skip_cut = atoi(value_ptr);
        if(skip_cut==0) { std::cout<<"Will keep runs not in the good runs list.\n\n"; }
        else if(skip_cut==1) { std::cout<<"Will remove runs not in the good runs list.\n\n"; }
        else{
          std::cout<<"Unknown -s option, setting to default skip_cut=1\n\n";
          skip_cut=1;
        }
      }
      break;

    case 'n':
      if (value_ptr) {
        flag_numi = atoi(value_ptr);
        if(flag_numi==0){ std::cout<<"Good runs list will be the one for BNB.\n\n"; }
        else if(flag_numi==1){ std::cout<<"Good runs list will be the one for NuMI.\n\n"; }
        else{
          std::cout<<"Unknown -n option, setting to default flag_numi=0\n\n";
          flag_numi=0;
        }
      }
      break;

    case 'c':
      if (value_ptr) {
        flag_data = atoi(value_ptr);
        if(flag_data==0){ std::cout<<"Good runs for list will be the one for MC.\n\n"; }
        else if(flag_data==1){ std::cout<<"Good runs for list will be the one for Data.\n\n"; }
        else{
          std::cout<<"Unknown -c option, setting to default flag_data=0\n\n";
          flag_data=0;
        }
      }
      break;

    case 'r':
      if (value_ptr) {
        remove_lantern_fails = atoi(value_ptr);
        if(remove_lantern_fails==0){ std::cout<<"Will Keep subruns where Lantern container failed.\n\n"; }
        else if(remove_lantern_fails==1){ std::cout<<"Removing subruns where Lantern container failed.\n\n"; }
        else{
          std::cout<<"Unknown -r option, setting to default remove_lantern_fails=1\n\n";
          remove_lantern_fails=1;
        }
      }
      break;

    case 'l':
      if (value_ptr) {
        training_list = value_ptr;
        std::cout<<"Loading Wire-Cell training list from "<<training_list<<"\n\n";
      }
      break;

    case 'i':
      if (value_ptr){
        remove_individual_run_list = value_ptr;
        std::cout<<"Loading list of additionally runs to remove from "<<remove_individual_run_list<<"\n\n";
      }
      break;

    case 'g':
      if (value_ptr) {
        global_file_type = value_ptr;
        std::cout<<"Setting Wire-Cell BDT training file type to "<<global_file_type<<"\n\n";
      }
      break;

    case 'b':
      if (value_ptr) {
        flag_keep_only_bdt_train = atoi(value_ptr);
        if(flag_keep_only_bdt_train==-1){ std::cout<<"Saving all runs regardless of Wire-Cell BDT training status.\n\n"; }
        else if(flag_keep_only_bdt_train==0){ std::cout<<"Only saving subruns that were not used for Wire-Cell BDT training.\n\n"; }
        else if(flag_keep_only_bdt_train==1){ std::cout<<"Only saving subruns that WERE used for Wire-Cell BDT training.\n\n"; }
        else{
          std::cout<<"Unknown -b option, setting to default flag_keep_only_bdt_train=-1\n\n";
          flag_keep_only_bdt_train=-1;
        }
      }
      break;

    case 'a':
      if (value_ptr) {
        flag_set_samdef = 1;
        samdef = value_ptr;
        std::cout<<"Saving the following samdef to trees: "<<samdef<<"\n\n";
      } else {
        std::cerr << "Missing value for -a" << std::endl;
      }
      break;

    case 'v':
      if (value_ptr && atoi(value_ptr) > 0) {
        set_verbose = atoi(value_ptr);
      } else {
        std::cout<<"Bad -v option, must be greater than 0. Leaving at 10000.\n\n";
      }
      break;

    case 'u':
      if (value_ptr && atoi(value_ptr) > 0) {
        set_verbose_pot = atoi(value_ptr);
      } else {
        std::cout<<"Bad -u option, must be greater than 0. Leaving at 1000.\n\n";
      }
      break;

    }

  }

  if( (training_list=="" || global_file_type=="") && flag_keep_only_bdt_train>=0){
    std::cout<<"Have flag_keep_only_bdt_train>=0, but no file list of file type set."<<std::endl; 
    std::cout<<"Please set -b-1 or add a file list and file type with -l and -g"<<std::endl;
    std::cout<<"Exiting."<<std::endl;
    return 1;
  }


  // Initiate the tree wranglers
  tree_wrangler wrangler(true, config_file_name, delimiter);
  if(flag_set_samdef) wrangler.set_samdef(flag_set_samdef, samdef);
  tree_wrangler wrangler_ex(true, config_file_name, delimiter,2);
  if(flag_set_samdef) wrangler_ex.set_samdef(flag_set_samdef, samdef);
  tree_wrangler wrangler_pot(true, config_file_name, delimiter,1);


  TFile *file1 = new TFile(input_file);


  // Load other trees from directories as specified by the config file
  wrangler.get_old_trees(file1);
  wrangler_ex.get_old_trees(file1);
  wrangler_pot.get_old_trees(file1);


  // Load good runs lists

  std::vector<int>good_run_list_vec = get_good_run_list();
  std::set<int> good_runlist_set(good_run_list_vec.begin(), good_run_list_vec.end());

  std::vector<int> low_lifetime_runs = get_low_lifetime_runs();
  std::set<int> low_lifetime_set(low_lifetime_runs.begin(), low_lifetime_runs.end());

  std::vector<int> low_neutrino_count_numi_run2RHC = get_low_neutrino_count_numi_run2RHC();
  std::set<int> low_neutrino_count_numi_run2RHC_set(low_neutrino_count_numi_run2RHC.begin(), low_neutrino_count_numi_run2RHC.end());


  // Load the BDT subruns if that flag is set
  std::map<string, std::set<std::pair<int, int> > > map_type_run_subrun;
  if(flag_keep_only_bdt_train>=0){
    if (training_list != ""){
      ifstream infile(training_list);
      if (!infile.good()) {
        std::cout<<"Unable to open Wire-Cell BDT training list. Exiting"<<std::endl;
        return 1;
      }
      string tmp_type;
      int run, subrun;
      while(infile >> tmp_type >> run >> subrun){
        map_type_run_subrun[tmp_type].insert(std::make_pair(run, subrun));
      }
    }
  }

  // Load list of individual runs to remove
  std::unordered_map<int,std::vector<int>> remove_individual_run;
  if (remove_individual_run_list != ""){
    ifstream infile(remove_individual_run_list);
    if (!infile.good()) {
      std::cout<<"Unable to open list of individual runs to remove. Exiting"<<std::endl;
      return 1;
    }
    int run;
    std::vector<int> subrun;
    std::string lineContent;
    while(std::getline(infile, lineContent)){
      run=-1;
      std::stringstream ss(lineContent);
      int entry;
      // Extract each subrun entry from the given line
      while (ss >> entry) {
        if(run<0) { run = entry; }
        else{
          subrun.push_back(entry);
        }
      }
      remove_individual_run[run] = subrun;
    }
  }

  // Check if the output file exists if overwrite is not set.
  if(flag_overwrite!=1){
    TFile *f = nullptr;
    if (!gSystem->AccessPathName(out_file)) TFile::Open(out_file, "READ");
    if (f && !f->IsZombie()) {
        std::cout<<'\n'<< "File exists. Exiting." << std::endl;
        f->Close();
        std::cout<<"Outputfile file "<<out_file<<" already exists and overwrite not set."<<std::endl;
        std::cout<<"Pick a new file name or use -o to force and overwrite of existing file."<<'\n'<<std::endl;
        return 1;
    }
  }


  TFile *file2 = new TFile(out_file,"RECREATE");


  //Setup the directories specified in the config file
  wrangler.set_new_trees(file2);
  wrangler_ex.set_new_trees(file2);
  wrangler_pot.set_new_trees(file2);


  // Build the pairs of pot trees
  wrangler_pot.grow_pot_arboretum();

  // Map out the relation between index and run-subrun for POT trees
  wrangler_pot.map_rs_to_entry();


  // Load Lantern, used when dropping subruns where container failed.
  int haveReco;
  TTree *T_lantern = (TTree*)file1->Get("lantern/EventTree");
  if(T_lantern && remove_lantern_fails==1) T_lantern->SetBranchAddress("haveReco",&haveReco);
  if(!T_lantern && remove_lantern_fails==1){
    std::cout<<"WARNING: remove_lantern_fails==1, but Lantern tree not found."<<'\n'<<std::endl;
  }

  // Figure out which tree you can load for the rse map
  int run;
  int subrun;
  int event;
  TTree *T_rse = nullptr;
  int found_rse_tree = get_T_rse(file1, T_rse, run, subrun, event);
  if(!found_rse_tree){
    std::cout<<'\n'<<"Could not find RSE tree. Exiting."<<std::endl;
    return 1;
  }

  int nentry = T_rse->GetEntries();
  if(start_events>nentry){
    std::cout<<'\n'<<"start_events>T_rse->GetEntries(). Exiting."<<std::endl;
    return 1;
  }


  // List of runs where Lantern container failed.
  std::unordered_set<std::pair<int,int>, PairHash> lantern_fail;


  // Create run-subrun-index map and exlude and bad subruns.
  // Key is run, subrun pair, each holds a pair, first being if its flagged as a good run, second being a vecotr of pairs of event,tree index.
  std::unordered_map<Key, Value, PairHash> run_sub_event_entry;

  std::cout<<"Starting first pass loop to order run-subruns. Will pass over "<<nentry<<" entries."<<std::endl;

  int verbose_counter=0;
  int event_counter=0;

  for (Int_t i=0;i!=nentry;i++){

    if ((i-verbose_counter)%set_verbose == 0) {
      std::cout << "    seen: "<<i<<"    passed: "  << event_counter << std::endl;
      verbose_counter=(int(int(i)/int(set_verbose)))*set_verbose;
    }

    T_rse->GetEntry(i);

    std::pair<int, int> run_subrun_pair = std::make_pair(run,subrun);

    // Already seen this run-subrun pair, no need to re-check it.
    if(run_sub_event_entry.find(run_subrun_pair)!=run_sub_event_entry.end()){
      auto& event_index_map = run_sub_event_entry[run_subrun_pair].second;
      // Event already exists, duplicate so append the index.
      if(event_index_map.find(event)!=event_index_map.end()){   
        std::cout<<"Found duplicate event: run,subrun,event = "<<run<<", "<<subrun<<", "<<event<<std::endl;
        if(flag_kill_duplicates==2){
          std::cout<<"Exiting. Can overide this by re-running with -k0"<<std::endl;
          return 1;
        }
        run_sub_event_entry[run_subrun_pair].second[event].push_back(i);
        if(run_sub_event_entry[run_subrun_pair].first) event_counter++;
        continue;
      }        
      std::vector<int> index_vector = {i};
      run_sub_event_entry[run_subrun_pair].second[event] = index_vector;
      if(run_sub_event_entry[run_subrun_pair].first) event_counter++;
      continue;
    }

    //                                      std::vector<int>                                                                                  indcies = {i};
    //               std::unordered_map<int,std::vector<int>>                                                       event_index_map = {{event,indcies}};
    //std::pair<bool,std::unordered_map<int,std::vector<int> >> pair_goodrun_event_index_map = std::make_pair(false,event_index_map);

    if(T_lantern) T_lantern->GetEntry(i);
    else haveReco=1;
    if (haveReco==0){
      lantern_fail.insert(std::make_pair(run,subrun));
    }
    bool temp_keep_subrun = keep_subrun(run, subrun, 
                                        start_run, start_subrun, stop_run, stop_subrun,
                                        remove_lantern_fails, haveReco, 
                                        flag_keep_only_bdt_train, global_file_type, map_type_run_subrun, 
                                        remove_individual_run, 
                                        skip_cut, flag_data, flag_numi,
                                        good_runlist_set, low_lifetime_set, low_neutrino_count_numi_run2RHC_set);

    auto& entry = run_sub_event_entry[run_subrun_pair];
    entry.first = temp_keep_subrun;
    entry.second[event].push_back(i);

    if(temp_keep_subrun) event_counter++;

  }
  
  std::cout << "    seen: "<<nentry<<"    passed: "  << event_counter << std::endl;


  // Turn the map into an ordered map
  std::map<Key, Value> run_sub_event_entry_first_subrun(run_sub_event_entry.begin(), run_sub_event_entry.end());


  // Copy all the event level trees to the new file. 

  std::cout<<'\n'<<'\n'<<"Begin looping over "<<nentry<<" events to fill trees."<<std::endl;

  int first_run=std::numeric_limits<int>::max();
  int first_subrun=std::numeric_limits<int>::max();
  int last_run=0;
  int last_subrun=0;

  int index_counter=0;
  event_counter=0;
  verbose_counter=0;

  for (auto rs_it = run_sub_event_entry_first_subrun.begin(); rs_it != run_sub_event_entry_first_subrun.end(); rs_it++){

    if ((index_counter-verbose_counter)%set_verbose == 0) {
      std::cout << "    seen: "<<index_counter<<"    saved: "  << event_counter<< std::endl;
      verbose_counter=(int(int(index_counter)/int(set_verbose)))*set_verbose;
    }

    int this_run = (*rs_it).first.first;
    int this_subrun = (*rs_it).first.second;
    bool good_subrun = (*rs_it).second.first;
    auto& event_index_map = (*rs_it).second.second;

    index_counter+=event_index_map.size();
    if(index_counter<=start_events) continue;

    // Skip event if the run-subrun got flagged as bad.
    if(!good_subrun){
      continue;
    }

    // Begin loop over events in this subrun
    for (auto e_it = event_index_map.begin(); e_it != event_index_map.end(); e_it++){

      int this_event = (*e_it).first;
      auto& index_vector = (*e_it).second; 
     
      // If allowing duplicates, loop over all duplicates of this event.
      for (int i_it=0; i_it<index_vector.size(); i_it++){
        if(flag_kill_duplicates==1 && i_it>0){
          std::cout<<"Removing Duplicat: run,subrun,event = "<<this_run<<", "<<this_subrun<<", "<<this_event<<"."<<std::endl;
          continue;
        }
        event_counter++;
        int this_index = index_vector.at(i_it);
        // Now fill all the trees.
        for(auto tree_it=wrangler.old_trees->begin(); tree_it!=wrangler.old_trees->end(); tree_it++){
          (*tree_it)->GetEntry(this_index);
        }
        for(auto tree_it=wrangler.new_trees->begin(); tree_it!=wrangler.new_trees->end(); tree_it++){
          (*tree_it)->Fill();
        }
        for(auto tree_it=wrangler_ex.old_trees->begin(); tree_it!=wrangler_ex.old_trees->end(); tree_it++){
          (*tree_it)->GetEntry(this_index);
        }
        for(auto tree_it=wrangler_ex.new_trees->begin(); tree_it!=wrangler_ex.new_trees->end(); tree_it++){
          (*tree_it)->Fill();
        }
      } // i_it, loop over duplicates of the event.

    } // e_it, loop over all events in the run-subrun.

    // Make sure you have the first and last run-subrun set as such
    if( (first_run>this_run) || (first_run==this_run && first_subrun>this_subrun) ){
      first_run=this_run;
      first_subrun=this_subrun;
    }
    if( (last_run<this_run) || (last_run==this_run && last_subrun<this_subrun) ){
      last_run=this_run;
      last_subrun=this_subrun;
    }

    // End the loop when you have enough events
    if(event_counter>=max_events){
      break;
    }

  } // rs_it, loop over all run-subruns

  std::cout << "    seen: "<<index_counter<<"    saved: "  << event_counter<< std::endl;


  // If saving the whole file overwrite limits, otherwise recover the events from the first subrun we started at if that is not complete.
  if(max_events>T_rse->GetEntries() && stop_run>last_run && start_events==0 && start_run==0){
    first_run=-1;
    first_subrun=-1;
    last_run=std::numeric_limits<int>::max();
    last_subrun=std::numeric_limits<int>::max();
  } 


  // Loop over each POT tree seperatly

  int rs_counter=0;

  for(auto pot_tree_it=wrangler_pot.pot_arboretum->begin(); pot_tree_it!=wrangler_pot.pot_arboretum->end(); pot_tree_it++){

    int nentry_pot = (*pot_tree_it)->old_pot_tree->GetEntries();
    std::cout<<'\n'<<"Begin looping over "<<(*pot_tree_it)->old_pot_tree->GetName()<<" tree with "<<nentry_pot<<" entries"<<std::endl;

    verbose_counter=0;
    rs_counter=0;

    for (Int_t i=0;i!=nentry_pot;i++){

      if ((i-verbose_counter)%set_verbose_pot == 0) {
        std::cout << "    seen: "<<i<<"    saved: "  << rs_counter<< std::endl;
        verbose_counter=(int(int(i)/int(set_verbose)))*set_verbose_pot;
      }

      (*pot_tree_it)->old_pot_tree->GetEntry(i);
      
      bool temp_haveReco=1;
      if (lantern_fail.find(std::make_pair((*pot_tree_it)->runNo,(*pot_tree_it)->subRunNo)) != lantern_fail.end()) {
        temp_haveReco=0;
      } 

      bool flag_keep_run =   keep_subrun((*pot_tree_it)->runNo, (*pot_tree_it)->subRunNo,
                                                         first_run, first_subrun, last_run, last_subrun, 
                                                         remove_lantern_fails, temp_haveReco, 
                                                         flag_keep_only_bdt_train, global_file_type, map_type_run_subrun,
                                                         remove_individual_run,
                                                         skip_cut, flag_data, flag_numi,
                                                         good_runlist_set, low_lifetime_set, low_neutrino_count_numi_run2RHC_set);

      if(!flag_keep_run) continue;

      (*pot_tree_it)->new_pot_tree->Fill();

      rs_counter+=1;

    }//i, loop over events in a given pot tree set

    std::cout << "    seen: "<<nentry_pot<<"    saved: "  << rs_counter << std::endl;

  }//pot_trees_it, loop over sets of pot trees


  file2->Write("",TFile::kOverwrite);
  file2->Close();


  std::cout<<'\n'<<"Saving output file: "<<out_file<<'\n'<<std::endl;


  return 0;


}
