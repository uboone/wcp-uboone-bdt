// cz: code modified from tutorials/tmva/TMVAClassification.C

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <set>
#include <unordered_map>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TMath.h"
#include "TKey.h"

#include "WCPLEEANA/tagger.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/Reader.h"


#include "WCPLEEANA/eval.h"

using namespace std;
using namespace LEEana;

#include "WCPLEEANA/bdt.h"
#include "WCPLEEANA/pot.h"
#include "WCPLEEANA/pfeval.h"
#include "WCPLEEANA/kine.h"
#include "WCPLEEANA/space.h"
#include "WCPLEEANA/particle.h"

#include "WCPLEEANA/tree_wrangler.h"


void print_help() {
  std::cout << R"(

========================================
 bdt_convert : Help
========================================

Overview:
---------
bdt_convert reads a Wire-Cell ROOT file and evaluates a large suite of
BDT-based classification scores for each event. It applies run/subrun-level
filtering, removes problematic subruns (e.g. Lantern failures or training
samples), enforces data-quality selections, and writes out a new ROOT file
with updated BDT variables and consistent POT handling.

The program:
  - Computes Wire-Cell BDT scores and related tagger variables
  - Removes subruns based on Lantern failures or training list selection
  - Applies data-quality and run-based filtering
  - Optionally overrides or rescales event weights
  - Copies additional trees using the tree_wrangler infrastructure
  - Produces a consistent output file with updated event-level information

Usage:
------
  bdt_convert <input_file> <output_file> [options]

Required arguments:
-------------------
  input_file     Input ROOT file
  output_file    Output ROOT file

Options:
--------

  -h
      Show this help message and exit

  -H
      Show help message for configuration file and exit

  -c<float>
      Maximum allowed weight (weight_cv * weight_spline)
      Events exceeding this are reset to weight = 1
      (default: 30)

  -f<float>
      (Reserved / legacy) failure percentage threshold
      (default: 0.15)

  -l<string>
      Path to Wire-Cell BDT training list file
      Format: <type> <run> <subrun>

  -i<string>
      Path to list of runs which will additionally be removed
      Format: <run> <subrun1> <subrun2> ...
      Use <run> -1 to remove all subruns in the run

  -g<string>
      Global file type label used with training list

  -b<int>
      Filter based on BDT training usage:
        0 = remove training subruns (default)
        1 = keep only training subruns

      NOTE: Requires -l (and optionally -g)

  -s<int>
      Skip data-quality cuts:
        0 = apply run quality cuts (default)
        1 = keep all runs

  -n<int>
      Beam selection:
        0 = BNB (default)
        1 = NuMI

  -r<int>
      Lantern failure handling:
        0 = keep subruns where Lantern failed
        1 = remove subruns where Lantern failed (default)

  -w<int>
      GiBUU mode:
        0 = normal handling (default)
        1 = override weights using truth timing information

  -p<int>
      Enable particle-level spacepoint BDTs:
        0 = disabled (default)
        1 = enabled

  -t<string>
      Configuration file for additional trees (default: config.txt)

  -d<char>
      Delimiter used in configuration file (default: ',')

  -a<string>
      Set SAM definition string to be stored in output trees

Configuration File:
-------------------
run bdt_convert -H for more info

Examples:
---------

  Basic usage:
    bdt_convert input.root output.root

  Apply stricter weight cut:
    bdt_convert input.root output.root -c20

  Remove training subruns:
    bdt_convert input.root output.root -ltrain.txt -b0

  Keep only training subruns:
    bdt_convert input.root output.root -ltrain.txt -b1

  Enable spacepoint BDTs:
    bdt_convert input.root output.root -p1

  Keep all runs (skip quality cuts):
    bdt_convert input.root output.root -s1

Notes:
------
- Event filtering is applied at the subrun level to maintain consistency.
- Subruns may be removed due to:
    * Lantern reconstruction failures
    * Presence (or absence) in BDT training lists
    * Data-quality run selections
- BDT scores are recomputed for every event and stored in the output trees.
- Weight handling protects against invalid or extreme values.
- Additional trees are copied using tree_wrangler configuration.

)";
}


int main( int argc, char** argv )
{
  if(argc==2 && argv[1][1]=='h'){
    print_help;
    return 0;
  }
  else if(argc==2 && argv[1][1]=='H'){
    print_help_wrangler_config(true);
    return 0;
  }
  else if (argc < 3) {
    std::cout << "bdt_convert #input_file #output_file -c[weight_cut_val] -l[traing_list] -g[global_file_type]" << std::endl;
    return -1;
  }

  TString input_file = argv[1];
  TString out_file = argv[2];

  float weight_cut_val = 30;
  float fail_percentage = 0.15;

  TString training_list = "";
  string global_file_type = "";
  TString remove_individual_run_list = "";
  int skip_cut = 0;
  int flag_numi = 0;

  bool flag_config = false;
  std::string config_file_name="config.txt";
  char delimiter = ',';

  bool flag_gibuu = false;

  bool flag_spbdt = false;

  int remove_lantern_fails = 1;

  int flag_keep_only_bdt_train = 0;

  int flag_set_samdef = 0;
  TString samdef="";

for (Int_t i = 3; i < argc; ++i) {

    // Skip non-flags
    if (argv[i][0] != '-') continue;

    char flag = argv[i][1];
    char* value_ptr = nullptr;

    // Case 1: attached value (-xVALUE)
    if (argv[i][2] != '\0') {
      value_ptr = &argv[i][2];
    }
    // Case 2: separate value (-x VALUE)
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

    case 'c':
      if (value_ptr) weight_cut_val = atof(value_ptr);
      break;

    case 'f':
      if (value_ptr) fail_percentage = atof(value_ptr);
      break;

    case 'l':
      if (value_ptr) training_list = value_ptr;
      break;

    case 'i':
      if (value_ptr) remove_individual_run_list = value_ptr;
      break;

    case 'g':
      if (value_ptr) global_file_type = value_ptr;
      break;

    case 's':
      if (value_ptr) skip_cut = atoi(value_ptr);
      break;

    case 'n':
      if (value_ptr) flag_numi = atoi(value_ptr);
      break;

    case 't':
      if (value_ptr) {
        config_file_name = value_ptr;
        flag_config = true;
      }
      break;

    case 'd':
      if (value_ptr) delimiter = value_ptr[0];
      break;

    case 'w':
      if (value_ptr) {
        flag_gibuu = atoi(value_ptr);
        if (flag_gibuu) {
          std::cout<<"GiBUU sample, overiding the weights"<<std::endl;
          std::cout<<std::endl;
        }
      }
      break;

    case 'p':
      if (value_ptr) flag_spbdt = atoi(value_ptr);
      break;

    case 'r':
      if (value_ptr) remove_lantern_fails = atoi(value_ptr);
      break;

    case 'b':
      if (value_ptr) flag_keep_only_bdt_train = atoi(value_ptr);
      break;

    case 'a':
      if (value_ptr) {
        flag_set_samdef = 1;
        samdef = value_ptr;
      } else {
        std::cerr << "Missing value for -a" << std::endl;
      }
      break;

    }

  }

  if (flag_spbdt) { 
    std::cout<<"Particle level spacepoint BDTs will be included"<<std::endl; 
    std::cout<<std::endl;
  }
  else {
    std::cout<<"No particle level spacepoint BDTs"<<std::endl; 
    std::cout<<std::endl;
  }

  if (remove_lantern_fails==1){
    std::cout<<"Removing subruns where Lantern container failed"<<std::endl;
    std::cout<<std::endl;
  } else {
    std::cout<<"Will keep subruns where Lantern container failed"<<std::endl;
    std::cout<<std::endl;
  }

  bool flag_check_run_subrun = false;
  bool flag_use_global_file_type = false;
  if (global_file_type != "") flag_use_global_file_type = true;

  std::map<string, std::set<std::pair<int, int> > > map_type_run_subrun;
  if (training_list != ""){
    flag_check_run_subrun = true;

    ifstream infile(training_list);
    string tmp_type;
    int run, subrun;
    while(!infile.eof()){
      infile >> tmp_type >> run >> subrun;
      map_type_run_subrun[tmp_type].insert(std::make_pair(run, subrun));
    }
    // std::cout << map_type_run_subrun.size() << std::endl;
    // return 0;
  }

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

  if (flag_keep_only_bdt_train) {
    std::cout<<"Only saving the run-subruns used to train Wire-Cell BDTs"<<std::endl; 
    if(flag_check_run_subrun==0) std::cout<<"WARNING, flag_check_run_subrun=false, so flag_keep_only_bdt_train has no effect"<<std::endl;
    std::cout<<std::endl;
  }

  if (skip_cut == 0) {
    std::cout << "Skip runs for BNB side " << std::endl; 
    std::cout<<std::endl;
  }
  else {
    std::cout << "Do not skip runs  " << std::endl;
    std::cout<<std::endl;
  }


  bool flag_data = true;

  tree_wrangler wrangler(flag_config, config_file_name, delimiter);
  if(flag_set_samdef) wrangler.set_samdef(flag_set_samdef, samdef);
  tree_wrangler wrangler_ex(flag_config, config_file_name, delimiter,2);
  if(flag_set_samdef) wrangler_ex.set_samdef(flag_set_samdef, samdef);
  tree_wrangler wrangler_pot(flag_config, config_file_name, delimiter,1);

  TFile *file1 = new TFile(input_file);
  TTree *T_BDTvars = (TTree*)file1->Get("wcpselection/T_BDTvars");

  TTree *T_eval = (TTree*)file1->Get("wcpselection/T_eval");
  TTree *T_pot = (TTree*)file1->Get("wcpselection/T_pot");
  TTree *T_PFeval = (TTree*)file1->Get("wcpselection/T_PFeval");
  TTree *T_KINEvars = (TTree*)file1->Get("wcpselection/T_KINEvars");
  TTree *T_spacepoints = (TTree*)file1->Get("wcpselection/T_spacepoints");
  TTree *T_lantern = (TTree*)file1->Get("lantern/EventTree");

  if (T_eval->GetBranch("weight_cv")) flag_data =false;


  //Load other trees from directories as specified by the config file
  wrangler.get_old_trees(file1);
  wrangler_ex.get_old_trees(file1);
  wrangler_pot.get_old_trees(file1);

  std::vector<int>good_run_list_vec = get_good_run_list();
  std::set<int> good_runlist_set(good_run_list_vec.begin(), good_run_list_vec.end());

  std::vector<int> low_lifetime_runs = get_low_lifetime_runs();
  std::set<int> low_lifetime_set(low_lifetime_runs.begin(), low_lifetime_runs.end());
  
  std::vector<int> low_neutrino_count_numi_run2RHC = get_low_neutrino_count_numi_run2RHC();
  std::set<int> low_neutrino_count_numi_run2RHC_set(low_neutrino_count_numi_run2RHC.begin(), low_neutrino_count_numi_run2RHC.end());
  
  
  TFile *file2 = new TFile(out_file,"RECREATE");

  //Setup the directories specified in the config file
  wrangler.set_new_trees(file2);
  wrangler_ex.set_new_trees(file2);
  wrangler_pot.set_new_trees(file2);

  // Build the pairs of pot trees
  wrangler_pot.grow_pot_arboretum();

  file2->mkdir("wcpselection");
  file2->cd("wcpselection");
  TTree *t4 = new TTree("T_BDTvars","T_BDTvars");
  TTree *t1 = new TTree("T_eval","T_eval");
  TTree *t2 = new TTree("T_pot","T_pot");
  TTree *t3 = new TTree("T_PFeval", "T_PFeval");
  TTree *t5 = new TTree("T_KINEvars", "T_KINEvars");
  TTree *new_T_spacepoints = T_spacepoints->CloneTree(0);


  EvalInfo eval;
  eval.file_type = new std::string();

  POTInfo pot;
  TaggerInfo tagger;
  PFevalInfo pfeval;
  KineInfo kine;

  SpaceInfo space_info;

  ParticleInfo particle_info;

  kine.kine_energy_particle = new std::vector<float>;
  kine.kine_energy_info = new std::vector<int>;
  kine.kine_particle_type = new std::vector<int>;
  kine.kine_energy_included = new std::vector<int>;


  tagger.pio_2_v_dis2 = new std::vector<float>;
  tagger.pio_2_v_angle2 = new std::vector<float>;
  tagger.pio_2_v_acc_length = new std::vector<float>;
  tagger.pio_2_v_flag = new std::vector<float>;
  tagger.sig_1_v_angle = new std::vector<float>;
  tagger.sig_1_v_flag_single_shower = new std::vector<float>;
  tagger.sig_1_v_energy = new std::vector<float>;
  tagger.sig_1_v_energy_1 = new std::vector<float>;
  tagger.sig_1_v_flag = new std::vector<float>;
  tagger.sig_2_v_energy = new std::vector<float>;
  tagger.sig_2_v_shower_angle = new std::vector<float>;
  tagger.sig_2_v_flag_single_shower = new std::vector<float>;
  tagger.sig_2_v_medium_dQ_dx = new std::vector<float>;
  tagger.sig_2_v_start_dQ_dx = new std::vector<float>;
  tagger.sig_2_v_flag = new std::vector<float>;
  tagger.stw_2_v_medium_dQ_dx = new std::vector<float>;
  tagger.stw_2_v_energy = new std::vector<float>;
  tagger.stw_2_v_angle = new std::vector<float>;
  tagger.stw_2_v_dir_length = new std::vector<float>;
  tagger.stw_2_v_max_dQ_dx = new std::vector<float>;
  tagger.stw_2_v_flag = new std::vector<float>;
  tagger.stw_3_v_angle = new std::vector<float>;
  tagger.stw_3_v_dir_length = new std::vector<float>;
  tagger.stw_3_v_energy = new std::vector<float>;
  tagger.stw_3_v_medium_dQ_dx = new std::vector<float>;
  tagger.stw_3_v_flag = new std::vector<float>;
  tagger.stw_4_v_angle = new std::vector<float>;
  tagger.stw_4_v_dis = new std::vector<float>;
  tagger.stw_4_v_energy = new std::vector<float>;
  tagger.stw_4_v_flag = new std::vector<float>;
  tagger.br3_3_v_energy = new std::vector<float>;
  tagger.br3_3_v_angle = new std::vector<float>;
  tagger.br3_3_v_dir_length = new std::vector<float>;
  tagger.br3_3_v_length = new std::vector<float>;
  tagger.br3_3_v_flag = new std::vector<float>;
  tagger.br3_5_v_dir_length = new std::vector<float>;
  tagger.br3_5_v_total_length = new std::vector<float>;
  tagger.br3_5_v_flag_avoid_muon_check = new std::vector<float>;
  tagger.br3_5_v_n_seg = new std::vector<float>;
  tagger.br3_5_v_angle = new std::vector<float>;
  tagger.br3_5_v_sg_length = new std::vector<float>;
  tagger.br3_5_v_energy = new std::vector<float>;
  tagger.br3_5_v_n_main_segs = new std::vector<float>;
  tagger.br3_5_v_n_segs = new std::vector<float>;
  tagger.br3_5_v_shower_main_length = new std::vector<float>;
  tagger.br3_5_v_shower_total_length = new std::vector<float>;
  tagger.br3_5_v_flag = new std::vector<float>;
  tagger.br3_6_v_angle = new std::vector<float>;
  tagger.br3_6_v_angle1 = new std::vector<float>;
  tagger.br3_6_v_flag_shower_trajectory = new std::vector<float>;
  tagger.br3_6_v_direct_length = new std::vector<float>;
  tagger.br3_6_v_length = new std::vector<float>;
  tagger.br3_6_v_n_other_vtx_segs = new std::vector<float>;
  tagger.br3_6_v_energy = new std::vector<float>;
  tagger.br3_6_v_flag = new std::vector<float>;
  tagger.tro_1_v_particle_type = new std::vector<float>;
  tagger.tro_1_v_flag_dir_weak = new std::vector<float>;
  tagger.tro_1_v_min_dis = new std::vector<float>;
  tagger.tro_1_v_sg1_length = new std::vector<float>;
  tagger.tro_1_v_shower_main_length = new std::vector<float>;
  tagger.tro_1_v_max_n_vtx_segs = new std::vector<float>;
  tagger.tro_1_v_tmp_length = new std::vector<float>;
  tagger.tro_1_v_medium_dQ_dx = new std::vector<float>;
  tagger.tro_1_v_dQ_dx_cut = new std::vector<float>;
  tagger.tro_1_v_flag_shower_topology = new std::vector<float>;
  tagger.tro_1_v_flag = new std::vector<float>;
  tagger.tro_2_v_energy = new std::vector<float>;
  tagger.tro_2_v_stem_length = new std::vector<float>;
  tagger.tro_2_v_iso_angle = new std::vector<float>;
  tagger.tro_2_v_max_length = new std::vector<float>;
  tagger.tro_2_v_angle = new std::vector<float>;
  tagger.tro_2_v_flag = new std::vector<float>;
  tagger.tro_4_v_dir2_mag = new std::vector<float>;
  tagger.tro_4_v_angle = new std::vector<float>;
  tagger.tro_4_v_angle1 = new std::vector<float>;
  tagger.tro_4_v_angle2 = new std::vector<float>;
  tagger.tro_4_v_length = new std::vector<float>;
  tagger.tro_4_v_length1 = new std::vector<float>;
  tagger.tro_4_v_medium_dQ_dx = new std::vector<float>;
  tagger.tro_4_v_end_dQ_dx = new std::vector<float>;
  tagger.tro_4_v_energy = new std::vector<float>;
  tagger.tro_4_v_shower_main_length = new std::vector<float>;
  tagger.tro_4_v_flag_shower_trajectory = new std::vector<float>;
  tagger.tro_4_v_flag = new std::vector<float>;
  tagger.tro_5_v_max_angle = new std::vector<float>;
  tagger.tro_5_v_min_angle = new std::vector<float>;
  tagger.tro_5_v_max_length = new std::vector<float>;
  tagger.tro_5_v_iso_angle = new std::vector<float>;
  tagger.tro_5_v_n_vtx_segs = new std::vector<float>;
  tagger.tro_5_v_min_count = new std::vector<float>;
  tagger.tro_5_v_max_count = new std::vector<float>;
  tagger.tro_5_v_energy = new std::vector<float>;
  tagger.tro_5_v_flag = new std::vector<float>;
  tagger.lol_1_v_energy = new std::vector<float>;
  tagger.lol_1_v_vtx_n_segs = new std::vector<float>;
  tagger.lol_1_v_nseg = new std::vector<float>;
  tagger.lol_1_v_angle = new std::vector<float>;
  tagger.lol_1_v_flag = new std::vector<float>;
  tagger.lol_2_v_length = new std::vector<float>;
  tagger.lol_2_v_angle = new std::vector<float>;
  tagger.lol_2_v_type = new std::vector<float>;
  tagger.lol_2_v_vtx_n_segs = new std::vector<float>;
  tagger.lol_2_v_energy = new std::vector<float>;
  tagger.lol_2_v_shower_main_length = new std::vector<float>;
  tagger.lol_2_v_flag_dir_weak = new std::vector<float>;
  tagger.lol_2_v_flag = new std::vector<float>;
  tagger.cosmict_flag_10 = new std::vector<float>;
  tagger.cosmict_10_flag_inside = new std::vector<float>;
  tagger.cosmict_10_vtx_z = new std::vector<float>;
  tagger.cosmict_10_flag_shower = new std::vector<float>;
  tagger.cosmict_10_flag_dir_weak = new std::vector<float>;
  tagger.cosmict_10_angle_beam = new std::vector<float>;
  tagger.cosmict_10_length = new std::vector<float>;
  tagger.numu_cc_flag_1 = new std::vector<float>;
  tagger.numu_cc_1_particle_type = new std::vector<float>;
  tagger.numu_cc_1_length = new std::vector<float>;
  tagger.numu_cc_1_medium_dQ_dx = new std::vector<float>;
  tagger.numu_cc_1_dQ_dx_cut = new std::vector<float>;
  tagger.numu_cc_1_direct_length = new std::vector<float>;
  tagger.numu_cc_1_n_daughter_tracks = new std::vector<float>;
  tagger.numu_cc_1_n_daughter_all = new std::vector<float>;
  tagger.numu_cc_flag_2 = new std::vector<float>;
  tagger.numu_cc_2_length = new std::vector<float>;
  tagger.numu_cc_2_total_length = new std::vector<float>;
  tagger.numu_cc_2_n_daughter_tracks = new std::vector<float>;
  tagger.numu_cc_2_n_daughter_all = new std::vector<float>;
  tagger.pio_2_v_dis2 = new std::vector<float>;
  tagger.pio_2_v_angle2 = new std::vector<float>;
  tagger.pio_2_v_acc_length = new std::vector<float>;
  tagger.pio_2_v_flag = new std::vector<float>;
  tagger.sig_1_v_angle = new std::vector<float>;
  tagger.sig_1_v_flag_single_shower = new std::vector<float>;
  tagger.sig_1_v_energy = new std::vector<float>;
  tagger.sig_1_v_energy_1 = new std::vector<float>;
  tagger.sig_1_v_flag = new std::vector<float>;
  tagger.sig_2_v_energy = new std::vector<float>;
  tagger.sig_2_v_shower_angle = new std::vector<float>;
  tagger.sig_2_v_flag_single_shower = new std::vector<float>;
  tagger.sig_2_v_medium_dQ_dx = new std::vector<float>;
  tagger.sig_2_v_start_dQ_dx = new std::vector<float>;
  tagger.sig_2_v_flag = new std::vector<float>;
  tagger.stw_2_v_medium_dQ_dx = new std::vector<float>;
  tagger.stw_2_v_energy = new std::vector<float>;
  tagger.stw_2_v_angle = new std::vector<float>;
  tagger.stw_2_v_dir_length = new std::vector<float>;
  tagger.stw_2_v_max_dQ_dx = new std::vector<float>;
  tagger.stw_2_v_flag = new std::vector<float>;
  tagger.stw_3_v_angle = new std::vector<float>;
  tagger.stw_3_v_dir_length = new std::vector<float>;
  tagger.stw_3_v_energy = new std::vector<float>;
  tagger.stw_3_v_medium_dQ_dx = new std::vector<float>;
  tagger.stw_3_v_flag = new std::vector<float>;
  tagger.stw_4_v_angle = new std::vector<float>;
  tagger.stw_4_v_dis = new std::vector<float>;
  tagger.stw_4_v_energy = new std::vector<float>;
  tagger.stw_4_v_flag = new std::vector<float>;
  tagger.br3_3_v_energy = new std::vector<float>;
  tagger.br3_3_v_angle = new std::vector<float>;
  tagger.br3_3_v_dir_length = new std::vector<float>;
  tagger.br3_3_v_length = new std::vector<float>;
  tagger.br3_3_v_flag = new std::vector<float>;
  tagger.br3_5_v_dir_length = new std::vector<float>;
  tagger.br3_5_v_total_length = new std::vector<float>;
  tagger.br3_5_v_flag_avoid_muon_check = new std::vector<float>;
  tagger.br3_5_v_n_seg = new std::vector<float>;
  tagger.br3_5_v_angle = new std::vector<float>;
  tagger.br3_5_v_sg_length = new std::vector<float>;
  tagger.br3_5_v_energy = new std::vector<float>;
  tagger.br3_5_v_n_main_segs = new std::vector<float>;
  tagger.br3_5_v_n_segs = new std::vector<float>;
  tagger.br3_5_v_shower_main_length = new std::vector<float>;
  tagger.br3_5_v_shower_total_length = new std::vector<float>;
  tagger.br3_5_v_flag = new std::vector<float>;
  tagger.br3_6_v_angle = new std::vector<float>;
  tagger.br3_6_v_angle1 = new std::vector<float>;
  tagger.br3_6_v_flag_shower_trajectory = new std::vector<float>;
  tagger.br3_6_v_direct_length = new std::vector<float>;
  tagger.br3_6_v_length = new std::vector<float>;
  tagger.br3_6_v_n_other_vtx_segs = new std::vector<float>;
  tagger.br3_6_v_energy = new std::vector<float>;
  tagger.br3_6_v_flag = new std::vector<float>;
  tagger.tro_1_v_particle_type = new std::vector<float>;
  tagger.tro_1_v_flag_dir_weak = new std::vector<float>;
  tagger.tro_1_v_min_dis = new std::vector<float>;
  tagger.tro_1_v_sg1_length = new std::vector<float>;
  tagger.tro_1_v_shower_main_length = new std::vector<float>;
  tagger.tro_1_v_max_n_vtx_segs = new std::vector<float>;
  tagger.tro_1_v_tmp_length = new std::vector<float>;
  tagger.tro_1_v_medium_dQ_dx = new std::vector<float>;
  tagger.tro_1_v_dQ_dx_cut = new std::vector<float>;
  tagger.tro_1_v_flag_shower_topology = new std::vector<float>;
  tagger.tro_1_v_flag = new std::vector<float>;
  tagger.tro_2_v_energy = new std::vector<float>;
  tagger.tro_2_v_stem_length = new std::vector<float>;
  tagger.tro_2_v_iso_angle = new std::vector<float>;
  tagger.tro_2_v_max_length = new std::vector<float>;
  tagger.tro_2_v_angle = new std::vector<float>;
  tagger.tro_2_v_flag = new std::vector<float>;
  tagger.tro_4_v_dir2_mag = new std::vector<float>;
  tagger.tro_4_v_angle = new std::vector<float>;
  tagger.tro_4_v_angle1 = new std::vector<float>;
  tagger.tro_4_v_angle2 = new std::vector<float>;
  tagger.tro_4_v_length = new std::vector<float>;
  tagger.tro_4_v_length1 = new std::vector<float>;
  tagger.tro_4_v_medium_dQ_dx = new std::vector<float>;
  tagger.tro_4_v_end_dQ_dx = new std::vector<float>;
  tagger.tro_4_v_energy = new std::vector<float>;
  tagger.tro_4_v_shower_main_length = new std::vector<float>;
  tagger.tro_4_v_flag_shower_trajectory = new std::vector<float>;
  tagger.tro_4_v_flag = new std::vector<float>;
  tagger.tro_5_v_max_angle = new std::vector<float>;
  tagger.tro_5_v_min_angle = new std::vector<float>;
  tagger.tro_5_v_max_length = new std::vector<float>;
  tagger.tro_5_v_iso_angle = new std::vector<float>;
  tagger.tro_5_v_n_vtx_segs = new std::vector<float>;
  tagger.tro_5_v_min_count = new std::vector<float>;
  tagger.tro_5_v_max_count = new std::vector<float>;
  tagger.tro_5_v_energy = new std::vector<float>;
  tagger.tro_5_v_flag = new std::vector<float>;
  tagger.lol_1_v_energy = new std::vector<float>;
  tagger.lol_1_v_vtx_n_segs = new std::vector<float>;
  tagger.lol_1_v_nseg = new std::vector<float>;
  tagger.lol_1_v_angle = new std::vector<float>;
  tagger.lol_1_v_flag = new std::vector<float>;
  tagger.lol_2_v_length = new std::vector<float>;
  tagger.lol_2_v_angle = new std::vector<float>;
  tagger.lol_2_v_type = new std::vector<float>;
  tagger.lol_2_v_vtx_n_segs = new std::vector<float>;
  tagger.lol_2_v_energy = new std::vector<float>;
  tagger.lol_2_v_shower_main_length = new std::vector<float>;
  tagger.lol_2_v_flag_dir_weak = new std::vector<float>;
  tagger.lol_2_v_flag = new std::vector<float>;
  tagger.cosmict_flag_10 = new std::vector<float>;
  tagger.cosmict_10_flag_inside = new std::vector<float>;
  tagger.cosmict_10_vtx_z = new std::vector<float>;
  tagger.cosmict_10_flag_shower = new std::vector<float>;
  tagger.cosmict_10_flag_dir_weak = new std::vector<float>;
  tagger.cosmict_10_angle_beam = new std::vector<float>;
  tagger.cosmict_10_length = new std::vector<float>;
  tagger.numu_cc_flag_1 = new std::vector<float>;
  tagger.numu_cc_1_particle_type = new std::vector<float>;
  tagger.numu_cc_1_length = new std::vector<float>;
  tagger.numu_cc_1_medium_dQ_dx = new std::vector<float>;
  tagger.numu_cc_1_dQ_dx_cut = new std::vector<float>;
  tagger.numu_cc_1_direct_length = new std::vector<float>;
  tagger.numu_cc_1_n_daughter_tracks = new std::vector<float>;
  tagger.numu_cc_1_n_daughter_all = new std::vector<float>;
  tagger.numu_cc_flag_2 = new std::vector<float>;
  tagger.numu_cc_2_length = new std::vector<float>;
  tagger.numu_cc_2_total_length = new std::vector<float>;
  tagger.numu_cc_2_n_daughter_tracks = new std::vector<float>;
  tagger.numu_cc_2_n_daughter_all = new std::vector<float>;
  tagger.ssm_kine_energy_particle = new std::vector<float>;
  tagger.ssm_kine_energy_info = new std::vector<int>;
  tagger.ssm_kine_particle_type = new std::vector<int>;
  tagger.ssm_kine_energy_included = new std::vector<int>;
  tagger.ssm_cosmict_flag_10 = new std::vector<float>;
  tagger.WCPMTInfoPePred = new std::vector<double>;
  tagger.WCPMTInfoPeMeas = new std::vector<double>;
  tagger.WCPMTInfoPeMeasErr = new std::vector<double>;

  particle_info.spacepoints_x = new std::vector<float>;
  particle_info.spacepoints_y = new std::vector<float>;
  particle_info.spacepoints_z = new std::vector<float>;
  particle_info.spacepoints_q = new std::vector<float>;

  set_tree_address(T_BDTvars, tagger,2 );
  tagger.flag_nc_gamma_bdt = true;
  tagger.flag_nc_gamma_0track_bdt = true;
  tagger.saved_ssm_bdt_scores = true;
  if(flag_spbdt) tagger.saved_pi_veto_scores = true;
  put_tree_address(t4, tagger,2);

  if (flag_data){
    set_tree_address(T_eval, eval,2);
    put_tree_address(t1, eval,2);

    set_tree_address(T_PFeval, pfeval,2);
    put_tree_address(t3, pfeval,2);
  }else{
    set_tree_address(T_eval, eval);
    put_tree_address(t1, eval);

    set_tree_address(T_PFeval, pfeval);
    put_tree_address(t3, pfeval);
  }

  set_tree_address(T_pot, pot);
  put_tree_address(t2, pot);



  set_tree_address(T_KINEvars, kine);
  put_tree_address(t5, kine);

  set_tree_address(T_spacepoints, space_info);
  //put_tree_address(new_T_spacepoints, space_info);

  //  bool match_isFC;
  //  T_eval->SetBranchAddress("match_isFC",&match_isFC);
  //  T_KINEvars->SetBranchAddress("kine_reco_Enu",&tagger.kine_reco_Enu);


  // BDTs ...
   // BDT stuff
  TMVA::Reader reader_br3_3;
  float br3_3_v_energy;
  float br3_3_v_angle;
  float br3_3_v_dir_length;
  float br3_3_v_length;
  reader_br3_3.AddVariable("br3_3_v_energy",&br3_3_v_energy);
  reader_br3_3.AddVariable("br3_3_v_angle",&br3_3_v_angle);
  reader_br3_3.AddVariable("br3_3_v_dir_length",&br3_3_v_dir_length);
  reader_br3_3.AddVariable("br3_3_v_length",&br3_3_v_length);
  reader_br3_3.BookMVA( "MyBDT", "weights/br3_3_BDT.weights.xml");

  TMVA::Reader reader_br3_5;
  float br3_5_v_dir_length;
  float br3_5_v_total_length;
  float br3_5_v_flag_avoid_muon_check;
  float br3_5_v_n_seg;
  float br3_5_v_angle;
  float br3_5_v_sg_length;
  float br3_5_v_energy;
  float br3_5_v_n_main_segs;
  float br3_5_v_n_segs;
  float br3_5_v_shower_main_length;
  float br3_5_v_shower_total_length;
  reader_br3_5.AddVariable("br3_5_v_dir_length",&br3_5_v_dir_length);
  reader_br3_5.AddVariable("br3_5_v_total_length",&br3_5_v_total_length);
  reader_br3_5.AddVariable("br3_5_v_flag_avoid_muon_check",&br3_5_v_flag_avoid_muon_check);
  reader_br3_5.AddVariable("br3_5_v_n_seg",&br3_5_v_n_seg);
  reader_br3_5.AddVariable("br3_5_v_angle",&br3_5_v_angle);
  reader_br3_5.AddVariable("br3_5_v_sg_length",&br3_5_v_sg_length);
  reader_br3_5.AddVariable("br3_5_v_energy",&br3_5_v_energy);
  reader_br3_5.AddVariable("br3_5_v_n_segs",&br3_5_v_n_segs);
  reader_br3_5.AddVariable("br3_5_v_shower_main_length",&br3_5_v_shower_main_length);
  reader_br3_5.AddVariable("br3_5_v_shower_total_length",&br3_5_v_shower_total_length);
  reader_br3_5.BookMVA( "MyBDT", "weights/br3_5_BDT.weights.xml");

  TMVA::Reader reader_br3_6;
  float br3_6_v_angle;
  float br3_6_v_angle1;
  float br3_6_v_flag_shower_trajectory;
  float br3_6_v_direct_length;
  float br3_6_v_length;
  float br3_6_v_n_other_vtx_segs;
  float br3_6_v_energy;
  reader_br3_6.AddVariable("br3_6_v_angle",&br3_6_v_angle);
  reader_br3_6.AddVariable("br3_6_v_angle1",&br3_6_v_angle1);
  reader_br3_6.AddVariable("br3_6_v_flag_shower_trajectory",&br3_6_v_flag_shower_trajectory);
  reader_br3_6.AddVariable("br3_6_v_direct_length",&br3_6_v_direct_length);
  reader_br3_6.AddVariable("br3_6_v_length",&br3_6_v_length);
  reader_br3_6.AddVariable("br3_6_v_n_other_vtx_segs",&br3_6_v_n_other_vtx_segs);
  reader_br3_6.AddVariable("br3_6_v_energy",&br3_6_v_energy);
  reader_br3_6.BookMVA( "MyBDT", "weights/br3_6_BDT.weights.xml");

  TMVA::Reader reader_pio_2;
  float pio_2_v_dis2;
  float pio_2_v_angle2;
  float pio_2_v_acc_length;
  reader_pio_2.AddVariable("pio_2_v_dis2",&pio_2_v_dis2);
  reader_pio_2.AddVariable("pio_2_v_angle2",&pio_2_v_angle2);
  reader_pio_2.AddVariable("pio_2_v_acc_length",&pio_2_v_acc_length);
  reader_pio_2.AddVariable("pio_mip_id",&tagger.pio_mip_id);
  reader_pio_2.BookMVA( "MyBDT", "weights/pio_2_BDT.weights.xml");

  TMVA::Reader reader_lol_1;
  float lol_1_v_energy;
  float lol_1_v_vtx_n_segs;
  float lol_1_v_nseg;
  float lol_1_v_angle;
  reader_lol_1.AddVariable("lol_1_v_energy",&lol_1_v_energy);
  reader_lol_1.AddVariable("lol_1_v_vtx_n_segs",&lol_1_v_vtx_n_segs);
  reader_lol_1.AddVariable("lol_1_v_nseg",&lol_1_v_nseg);
  reader_lol_1.AddVariable("lol_1_v_angle",&lol_1_v_angle);
  reader_lol_1.BookMVA( "MyBDT", "weights/lol_1_BDT.weights.xml");

  TMVA::Reader reader_lol_2;
  float lol_2_v_length;
  float lol_2_v_angle;
  float lol_2_v_type;
  float lol_2_v_vtx_n_segs;
  float lol_2_v_energy;
  float lol_2_v_shower_main_length;
  float lol_2_v_flag_dir_weak;
  reader_lol_2.AddVariable("lol_2_v_length",&lol_2_v_length);
  reader_lol_2.AddVariable("lol_2_v_angle",&lol_2_v_angle);
  reader_lol_2.AddVariable("lol_2_v_type",&lol_2_v_type);
  reader_lol_2.AddVariable("lol_2_v_vtx_n_segs",&lol_2_v_vtx_n_segs);
  reader_lol_2.AddVariable("lol_2_v_energy",&lol_2_v_energy);
  reader_lol_2.AddVariable("lol_2_v_shower_main_length",&lol_2_v_shower_main_length);
  reader_lol_2.AddVariable("lol_2_v_flag_dir_weak",&lol_2_v_flag_dir_weak);
  reader_lol_2.BookMVA( "MyBDT", "weights/lol_2_BDT.weights.xml");

  TMVA::Reader reader_tro_1;
  float tro_1_v_particle_type;
  float tro_1_v_flag_dir_weak;
  float tro_1_v_min_dis;
  float tro_1_v_sg1_length;
  float tro_1_v_shower_main_length;
  float tro_1_v_max_n_vtx_segs;
  float tro_1_v_tmp_length;
  float tro_1_v_medium_dQ_dx;
  float tro_1_v_dQ_dx_cut;
  float tro_1_v_flag_shower_topology;
  reader_tro_1.AddVariable("tro_1_v_particle_type",&tro_1_v_particle_type);
  reader_tro_1.AddVariable("tro_1_v_flag_dir_weak",&tro_1_v_flag_dir_weak);
  reader_tro_1.AddVariable("tro_1_v_min_dis",&tro_1_v_min_dis);
  reader_tro_1.AddVariable("tro_1_v_sg1_length",&tro_1_v_sg1_length);
  reader_tro_1.AddVariable("tro_1_v_shower_main_length",&tro_1_v_shower_main_length);
  reader_tro_1.AddVariable("tro_1_v_max_n_vtx_segs",&tro_1_v_max_n_vtx_segs);
  reader_tro_1.AddVariable("tro_1_v_tmp_length",&tro_1_v_tmp_length);
  reader_tro_1.AddVariable("tro_1_v_medium_dQ_dx",&tro_1_v_medium_dQ_dx);
  reader_tro_1.AddVariable("tro_1_v_dQ_dx_cut", &tro_1_v_dQ_dx_cut);
  reader_tro_1.AddVariable("tro_1_v_flag_shower_topology", &tro_1_v_flag_shower_topology);
  reader_tro_1.BookMVA( "MyBDT", "weights/tro_1_BDT.weights.xml");

  TMVA::Reader reader_tro_2;
  float tro_2_v_energy;
  float tro_2_v_stem_length;
  float tro_2_v_iso_angle;
  float tro_2_v_max_length;
  float tro_2_v_angle;
  reader_tro_2.AddVariable("tro_2_v_energy",&tro_2_v_energy);
  reader_tro_2.AddVariable("tro_2_v_stem_length",&tro_2_v_stem_length);
  reader_tro_2.AddVariable("tro_2_v_iso_angle",&tro_2_v_iso_angle);
  reader_tro_2.AddVariable("tro_2_v_max_length",&tro_2_v_max_length);
  reader_tro_2.AddVariable("tro_2_v_angle",&tro_2_v_angle);
  reader_tro_2.BookMVA( "MyBDT", "weights/tro_2_BDT.weights.xml");

  TMVA::Reader reader_tro_4;

  float tro_4_v_dir2_mag;
  float tro_4_v_angle;
  float tro_4_v_angle1;
  float tro_4_v_angle2;
  float tro_4_v_length;
  float tro_4_v_length1;
  float tro_4_v_medium_dQ_dx;
  float tro_4_v_end_dQ_dx;
  float tro_4_v_energy;
  float tro_4_v_shower_main_length;
  float tro_4_v_flag_shower_trajectory;

  reader_tro_4.AddVariable("tro_4_v_dir2_mag",&tro_4_v_dir2_mag);
  reader_tro_4.AddVariable("tro_4_v_angle",&tro_4_v_angle);
  reader_tro_4.AddVariable("tro_4_v_angle1",&tro_4_v_angle1);
  reader_tro_4.AddVariable("tro_4_v_angle2",&tro_4_v_angle2);
  reader_tro_4.AddVariable("tro_4_v_length",&tro_4_v_length);
  reader_tro_4.AddVariable("tro_4_v_length1",&tro_4_v_length1);
  reader_tro_4.AddVariable("tro_4_v_medium_dQ_dx",&tro_4_v_medium_dQ_dx);
  reader_tro_4.AddVariable("tro_4_v_end_dQ_dx",&tro_4_v_end_dQ_dx);
  reader_tro_4.AddVariable("tro_4_v_energy",&tro_4_v_energy);
  reader_tro_4.AddVariable("tro_4_v_shower_main_length",&tro_4_v_shower_main_length);
  reader_tro_4.AddVariable("tro_4_v_flag_shower_trajectory",&tro_4_v_flag_shower_trajectory);
  reader_tro_4.BookMVA( "MyBDT", "weights/tro_4_BDT.weights.xml");

  TMVA::Reader reader_tro_5;
  float tro_5_v_max_angle;
  float tro_5_v_min_angle;
  float tro_5_v_max_length;
  float tro_5_v_iso_angle;
  float tro_5_v_n_vtx_segs;
  float tro_5_v_min_count;
  float tro_5_v_max_count;
  float tro_5_v_energy;
  reader_tro_5.AddVariable("tro_5_v_max_angle",&tro_5_v_max_angle);
  reader_tro_5.AddVariable("tro_5_v_min_angle",&tro_5_v_min_angle);
  reader_tro_5.AddVariable("tro_5_v_max_length",&tro_5_v_max_length);
  reader_tro_5.AddVariable("tro_5_v_iso_angle",&tro_5_v_iso_angle);
  reader_tro_5.AddVariable("tro_5_v_n_vtx_segs",&tro_5_v_n_vtx_segs);
  reader_tro_5.AddVariable("tro_5_v_min_count",&tro_5_v_min_count);
  reader_tro_5.AddVariable("tro_5_v_max_count",&tro_5_v_max_count);
  reader_tro_5.AddVariable("tro_5_v_energy",&tro_5_v_energy);
  reader_tro_5.BookMVA( "MyBDT", "weights/tro_5_BDT.weights.xml");

  TMVA::Reader reader_sig_1;
  float sig_1_v_angle;
  float sig_1_v_flag_single_shower;
  float sig_1_v_energy;
  float sig_1_v_energy_1;
  reader_sig_1.AddVariable("sig_1_v_angle",&sig_1_v_angle);
  reader_sig_1.AddVariable("sig_1_v_flag_single_shower",&sig_1_v_flag_single_shower);
  reader_sig_1.AddVariable("sig_1_v_energy",&sig_1_v_energy);
  reader_sig_1.AddVariable("sig_1_v_energy_1",&sig_1_v_energy_1);
  reader_sig_1.BookMVA( "MyBDT", "weights/sig_1_BDT.weights.xml");

  TMVA::Reader reader_sig_2;
  float sig_2_v_energy;
  float sig_2_v_shower_angle;
  float sig_2_v_flag_single_shower;
  float sig_2_v_medium_dQ_dx;
  float sig_2_v_start_dQ_dx;

  reader_sig_2.AddVariable("sig_2_v_energy",&sig_2_v_energy);
  reader_sig_2.AddVariable("sig_2_v_shower_angle",&sig_2_v_shower_angle);
  reader_sig_2.AddVariable("sig_2_v_flag_single_shower",&sig_2_v_flag_single_shower);
  reader_sig_2.AddVariable("sig_2_v_medium_dQ_dx",&sig_2_v_medium_dQ_dx);
  reader_sig_2.AddVariable("sig_2_v_start_dQ_dx",&sig_2_v_start_dQ_dx);

  reader_sig_2.BookMVA( "MyBDT", "weights/sig_2_BDT.weights.xml");

  TMVA::Reader reader_stw_2;

  float stw_2_v_medium_dQ_dx;
  float stw_2_v_energy;
  float stw_2_v_angle;
  float stw_2_v_dir_length;
  float stw_2_v_max_dQ_dx;

  reader_stw_2.AddVariable("stw_2_v_medium_dQ_dx",&stw_2_v_medium_dQ_dx);
  reader_stw_2.AddVariable("stw_2_v_energy",&stw_2_v_energy);
  reader_stw_2.AddVariable("stw_2_v_angle",&stw_2_v_angle);
  reader_stw_2.AddVariable("stw_2_v_dir_length",&stw_2_v_dir_length);
  reader_stw_2.AddVariable("stw_2_v_max_dQ_dx",&stw_2_v_max_dQ_dx);

  reader_stw_2.BookMVA( "MyBDT", "weights/stw_2_BDT.weights.xml");

  TMVA::Reader reader_stw_3;

  float stw_3_v_angle;
  float stw_3_v_dir_length;
  float stw_3_v_energy;
  float stw_3_v_medium_dQ_dx;

  reader_stw_3.AddVariable("stw_3_v_angle",&stw_3_v_angle);
  reader_stw_3.AddVariable("stw_3_v_dir_length",&stw_3_v_dir_length);
  reader_stw_3.AddVariable("stw_3_v_energy",&stw_3_v_energy);
  reader_stw_3.AddVariable("stw_3_v_medium_dQ_dx",&stw_3_v_medium_dQ_dx);

  reader_stw_3.BookMVA( "MyBDT", "weights/stw_3_BDT.weights.xml");

  TMVA::Reader reader_stw_4;

  float stw_4_v_angle;
  float stw_4_v_dis;
  float stw_4_v_energy;

  reader_stw_4.AddVariable("stw_4_v_angle",&stw_4_v_angle);
  reader_stw_4.AddVariable("stw_4_v_dis",&stw_4_v_dis);
  reader_stw_4.AddVariable("stw_4_v_energy",&stw_4_v_energy);
  reader_stw_4.BookMVA( "MyBDT", "weights/stw_4_BDT.weights.xml");


  // total BDTs ... to be added ...
  TMVA::Reader reader;
  reader.AddVariable("match_isFC",&tagger.match_isFC);
  reader.AddVariable("kine_reco_Enu",&tagger.kine_reco_Enu);

  reader.AddVariable("cme_mu_energy",&tagger.cme_mu_energy);
  reader.AddVariable("cme_energy",&tagger.cme_energy);
  reader.AddVariable("cme_mu_length",&tagger.cme_mu_length);
  reader.AddVariable("cme_length",&tagger.cme_length);
  reader.AddVariable("cme_angle_beam",&tagger.cme_angle_beam);
  reader.AddVariable("anc_angle",&tagger.anc_angle);
  reader.AddVariable("anc_max_angle",&tagger.anc_max_angle);
  reader.AddVariable("anc_max_length",&tagger.anc_max_length);
  reader.AddVariable("anc_acc_forward_length",&tagger.anc_acc_forward_length);
  reader.AddVariable("anc_acc_backward_length",&tagger.anc_acc_backward_length);
  reader.AddVariable("anc_acc_forward_length1",&tagger.anc_acc_forward_length1);
  reader.AddVariable("anc_shower_main_length",&tagger.anc_shower_main_length);
  reader.AddVariable("anc_shower_total_length",&tagger.anc_shower_total_length);
  reader.AddVariable("anc_flag_main_outside",&tagger.anc_flag_main_outside);
  reader.AddVariable("gap_flag_prolong_u",&tagger.gap_flag_prolong_u);
  reader.AddVariable("gap_flag_prolong_v",&tagger.gap_flag_prolong_v);
  reader.AddVariable("gap_flag_prolong_w",&tagger.gap_flag_prolong_w);
  reader.AddVariable("gap_flag_parallel",&tagger.gap_flag_parallel);
  reader.AddVariable("gap_n_points",&tagger.gap_n_points);
  reader.AddVariable("gap_n_bad",&tagger.gap_n_bad);
  reader.AddVariable("gap_energy",&tagger.gap_energy);
  reader.AddVariable("gap_num_valid_tracks",&tagger.gap_num_valid_tracks);
  reader.AddVariable("gap_flag_single_shower",&tagger.gap_flag_single_shower);
  reader.AddVariable("hol_1_n_valid_tracks",&tagger.hol_1_n_valid_tracks);
  reader.AddVariable("hol_1_min_angle",&tagger.hol_1_min_angle);
  reader.AddVariable("hol_1_energy",&tagger.hol_1_energy);
  reader.AddVariable("hol_1_min_length",&tagger.hol_1_min_length);
  reader.AddVariable("hol_2_min_angle",&tagger.hol_2_min_angle);
  reader.AddVariable("hol_2_medium_dQ_dx",&tagger.hol_2_medium_dQ_dx);
  reader.AddVariable("hol_2_ncount",&tagger.hol_2_ncount);
  reader.AddVariable("lol_3_angle_beam",&tagger.lol_3_angle_beam);
  reader.AddVariable("lol_3_n_valid_tracks",&tagger.lol_3_n_valid_tracks);
  reader.AddVariable("lol_3_min_angle",&tagger.lol_3_min_angle);
  reader.AddVariable("lol_3_vtx_n_segs",&tagger.lol_3_vtx_n_segs);
  reader.AddVariable("lol_3_shower_main_length",&tagger.lol_3_shower_main_length);
  reader.AddVariable("lol_3_n_out",&tagger.lol_3_n_out);
  reader.AddVariable("lol_3_n_sum",&tagger.lol_3_n_sum);
  reader.AddVariable("hol_1_flag_all_shower",&tagger.hol_1_flag_all_shower); // naming issue
  reader.AddVariable("mgo_energy",&tagger.mgo_energy);
  reader.AddVariable("mgo_max_energy",&tagger.mgo_max_energy);
  reader.AddVariable("mgo_total_energy",&tagger.mgo_total_energy);
  reader.AddVariable("mgo_n_showers",&tagger.mgo_n_showers);
  reader.AddVariable("mgo_max_energy_1",&tagger.mgo_max_energy_1);
  reader.AddVariable("mgo_max_energy_2",&tagger.mgo_max_energy_2);
  reader.AddVariable("mgo_total_other_energy",&tagger.mgo_total_other_energy);
  reader.AddVariable("mgo_n_total_showers",&tagger.mgo_n_total_showers);
  reader.AddVariable("mgo_total_other_energy_1",&tagger.mgo_total_other_energy_1);
  reader.AddVariable("mgt_flag_single_shower",&tagger.mgt_flag_single_shower);
  reader.AddVariable("mgt_max_energy",&tagger.mgt_max_energy);
  reader.AddVariable("mgt_total_other_energy",&tagger.mgt_total_other_energy);
  reader.AddVariable("mgt_max_energy_1",&tagger.mgt_max_energy_1);
  reader.AddVariable("mgt_e_indirect_max_energy",&tagger.mgt_e_indirect_max_energy);
  reader.AddVariable("mgt_e_direct_max_energy",&tagger.mgt_e_direct_max_energy);
  reader.AddVariable("mgt_n_direct_showers",&tagger.mgt_n_direct_showers);
  reader.AddVariable("mgt_e_direct_total_energy",&tagger.mgt_e_direct_total_energy);
  reader.AddVariable("mgt_flag_indirect_max_pio",&tagger.mgt_flag_indirect_max_pio);
  reader.AddVariable("mgt_e_indirect_total_energy",&tagger.mgt_e_indirect_total_energy);
  reader.AddVariable("mip_quality_energy",&tagger.mip_quality_energy);
  reader.AddVariable("mip_quality_overlap",&tagger.mip_quality_overlap);
  reader.AddVariable("mip_quality_n_showers",&tagger.mip_quality_n_showers);
  reader.AddVariable("mip_quality_n_tracks",&tagger.mip_quality_n_tracks);
  reader.AddVariable("mip_quality_flag_inside_pi0",&tagger.mip_quality_flag_inside_pi0);
  reader.AddVariable("mip_quality_n_pi0_showers",&tagger.mip_quality_n_pi0_showers);
  reader.AddVariable("mip_quality_shortest_length",&tagger.mip_quality_shortest_length);
  reader.AddVariable("mip_quality_acc_length",&tagger.mip_quality_acc_length);
  reader.AddVariable("mip_quality_shortest_angle",&tagger.mip_quality_shortest_angle);
  reader.AddVariable("mip_quality_flag_proton",&tagger.mip_quality_flag_proton);
  reader.AddVariable("br1_1_shower_type",&tagger.br1_1_shower_type);
  reader.AddVariable("br1_1_vtx_n_segs",&tagger.br1_1_vtx_n_segs);
  reader.AddVariable("br1_1_energy",&tagger.br1_1_energy);
  reader.AddVariable("br1_1_n_segs",&tagger.br1_1_n_segs);
  reader.AddVariable("br1_1_flag_sg_topology",&tagger.br1_1_flag_sg_topology);
  reader.AddVariable("br1_1_flag_sg_trajectory",&tagger.br1_1_flag_sg_trajectory);
  reader.AddVariable("br1_1_sg_length",&tagger.br1_1_sg_length);
  reader.AddVariable("br1_2_n_connected",&tagger.br1_2_n_connected);
  reader.AddVariable("br1_2_max_length",&tagger.br1_2_max_length);
  reader.AddVariable("br1_2_n_connected_1",&tagger.br1_2_n_connected_1);
  reader.AddVariable("br1_2_n_shower_segs",&tagger.br1_2_n_shower_segs);
  reader.AddVariable("br1_2_max_length_ratio",&tagger.br1_2_max_length_ratio);
  reader.AddVariable("br1_2_shower_length",&tagger.br1_2_shower_length);
  reader.AddVariable("br1_3_n_connected_p",&tagger.br1_3_n_connected_p);
  reader.AddVariable("br1_3_max_length_p",&tagger.br1_3_max_length_p);
  reader.AddVariable("br1_3_n_shower_main_segs",&tagger.br1_3_n_shower_main_segs);
  reader.AddVariable("br3_1_energy",&tagger.br3_1_energy);
  reader.AddVariable("br3_1_n_shower_segments",&tagger.br3_1_n_shower_segments);
  reader.AddVariable("br3_1_sg_flag_trajectory",&tagger.br3_1_sg_flag_trajectory);
  reader.AddVariable("br3_1_sg_direct_length",&tagger.br3_1_sg_direct_length);
  reader.AddVariable("br3_1_sg_length",&tagger.br3_1_sg_length);
  reader.AddVariable("br3_1_total_main_length",&tagger.br3_1_total_main_length);
  reader.AddVariable("br3_1_total_length",&tagger.br3_1_total_length);
  reader.AddVariable("br3_1_iso_angle",&tagger.br3_1_iso_angle);
  reader.AddVariable("br3_1_sg_flag_topology",&tagger.br3_1_sg_flag_topology);
  reader.AddVariable("br3_2_n_ele",&tagger.br3_2_n_ele);
  reader.AddVariable("br3_2_n_other",&tagger.br3_2_n_other);
  reader.AddVariable("br3_2_other_fid",&tagger.br3_2_other_fid);
  reader.AddVariable("br3_4_acc_length",&tagger.br3_4_acc_length);
  reader.AddVariable("br3_4_total_length",&tagger.br3_4_total_length);
  reader.AddVariable("br3_7_min_angle",&tagger.br3_7_min_angle);
  reader.AddVariable("br3_8_max_dQ_dx",&tagger.br3_8_max_dQ_dx);
  reader.AddVariable("br3_8_n_main_segs",&tagger.br3_8_n_main_segs);
  reader.AddVariable("vis_1_n_vtx_segs",&tagger.vis_1_n_vtx_segs);
  reader.AddVariable("vis_1_energy",&tagger.vis_1_energy);
  reader.AddVariable("vis_1_num_good_tracks",&tagger.vis_1_num_good_tracks);
  reader.AddVariable("vis_1_max_angle",&tagger.vis_1_max_angle);
  reader.AddVariable("vis_1_max_shower_angle",&tagger.vis_1_max_shower_angle);
  reader.AddVariable("vis_1_tmp_length1",&tagger.vis_1_tmp_length1);
  reader.AddVariable("vis_1_tmp_length2",&tagger.vis_1_tmp_length2);
  reader.AddVariable("vis_2_n_vtx_segs",&tagger.vis_2_n_vtx_segs);
  reader.AddVariable("vis_2_min_angle",&tagger.vis_2_min_angle);
  reader.AddVariable("vis_2_min_weak_track",&tagger.vis_2_min_weak_track);
  reader.AddVariable("vis_2_angle_beam",&tagger.vis_2_angle_beam);
  reader.AddVariable("vis_2_min_angle1",&tagger.vis_2_min_angle1);
  reader.AddVariable("vis_2_iso_angle1",&tagger.vis_2_iso_angle1);
  reader.AddVariable("vis_2_min_medium_dQ_dx",&tagger.vis_2_min_medium_dQ_dx);
  reader.AddVariable("vis_2_min_length",&tagger.vis_2_min_length);
  reader.AddVariable("vis_2_sg_length",&tagger.vis_2_sg_length);
  reader.AddVariable("vis_2_max_angle",&tagger.vis_2_max_angle);
  reader.AddVariable("vis_2_max_weak_track",&tagger.vis_2_max_weak_track);
  reader.AddVariable("pio_1_mass",&tagger.pio_1_mass);
  reader.AddVariable("pio_1_pio_type",&tagger.pio_1_pio_type);
  reader.AddVariable("pio_1_energy_1",&tagger.pio_1_energy_1);
  reader.AddVariable("pio_1_energy_2",&tagger.pio_1_energy_2);
  reader.AddVariable("pio_1_dis_1",&tagger.pio_1_dis_1);
  reader.AddVariable("pio_1_dis_2",&tagger.pio_1_dis_2);
  reader.AddVariable("pio_mip_id",&tagger.pio_mip_id);
  reader.AddVariable("stem_dir_flag_single_shower",&tagger.stem_dir_flag_single_shower);
  reader.AddVariable("stem_dir_angle",&tagger.stem_dir_angle);
  reader.AddVariable("stem_dir_energy",&tagger.stem_dir_energy);
  reader.AddVariable("stem_dir_angle1",&tagger.stem_dir_angle1);
  reader.AddVariable("stem_dir_angle2",&tagger.stem_dir_angle2);
  reader.AddVariable("stem_dir_angle3",&tagger.stem_dir_angle3);
  reader.AddVariable("stem_dir_ratio",&tagger.stem_dir_ratio);
  reader.AddVariable("br2_num_valid_tracks",&tagger.br2_num_valid_tracks);
  reader.AddVariable("br2_n_shower_main_segs",&tagger.br2_n_shower_main_segs);
  reader.AddVariable("br2_max_angle",&tagger.br2_max_angle);
  reader.AddVariable("br2_sg_length",&tagger.br2_sg_length);
  reader.AddVariable("br2_flag_sg_trajectory",&tagger.br2_flag_sg_trajectory);
  reader.AddVariable("stem_len_energy",&tagger.stem_len_energy);
  reader.AddVariable("stem_len_length",&tagger.stem_len_length);
  reader.AddVariable("stem_len_flag_avoid_muon_check",&tagger.stem_len_flag_avoid_muon_check);
  reader.AddVariable("stem_len_num_daughters",&tagger.stem_len_num_daughters);
  reader.AddVariable("stem_len_daughter_length",&tagger.stem_len_daughter_length);
  reader.AddVariable("brm_n_mu_segs",&tagger.brm_n_mu_segs);
  reader.AddVariable("brm_Ep",&tagger.brm_Ep);
  reader.AddVariable("brm_acc_length",&tagger.brm_acc_length);
  reader.AddVariable("brm_shower_total_length",&tagger.brm_shower_total_length);
  reader.AddVariable("brm_connected_length",&tagger.brm_connected_length);
  reader.AddVariable("brm_n_size",&tagger.brm_n_size);
  reader.AddVariable("brm_n_shower_main_segs",&tagger.brm_n_shower_main_segs);
  reader.AddVariable("brm_n_mu_main",&tagger.brm_n_mu_main);
  reader.AddVariable("lem_shower_main_length",&tagger.lem_shower_main_length);
  reader.AddVariable("lem_n_3seg",&tagger.lem_n_3seg);
  reader.AddVariable("lem_e_charge",&tagger.lem_e_charge);
  reader.AddVariable("lem_e_dQdx",&tagger.lem_e_dQdx);
  reader.AddVariable("lem_shower_num_main_segs",&tagger.lem_shower_num_main_segs);
  reader.AddVariable("brm_acc_direct_length",&tagger.brm_acc_direct_length); // naming issue
  reader.AddVariable("stw_1_energy",&tagger.stw_1_energy);
  reader.AddVariable("stw_1_dis",&tagger.stw_1_dis);
  reader.AddVariable("stw_1_dQ_dx",&tagger.stw_1_dQ_dx);
  reader.AddVariable("stw_1_flag_single_shower",&tagger.stw_1_flag_single_shower);
  reader.AddVariable("stw_1_n_pi0",&tagger.stw_1_n_pi0);
  reader.AddVariable("stw_1_num_valid_tracks",&tagger.stw_1_num_valid_tracks);
  reader.AddVariable("spt_shower_main_length",&tagger.spt_shower_main_length);
  reader.AddVariable("spt_shower_total_length",&tagger.spt_shower_total_length);
  reader.AddVariable("spt_angle_beam",&tagger.spt_angle_beam);
  reader.AddVariable("spt_angle_vertical",&tagger.spt_angle_vertical);
  reader.AddVariable("spt_max_dQ_dx",&tagger.spt_max_dQ_dx);
  reader.AddVariable("spt_angle_beam_1",&tagger.spt_angle_beam_1);
  reader.AddVariable("spt_angle_drift",&tagger.spt_angle_drift);
  reader.AddVariable("spt_angle_drift_1",&tagger.spt_angle_drift_1);
  reader.AddVariable("spt_num_valid_tracks",&tagger.spt_num_valid_tracks);
  reader.AddVariable("spt_n_vtx_segs",&tagger.spt_n_vtx_segs);
  reader.AddVariable("spt_max_length",&tagger.spt_max_length);
  reader.AddVariable("mip_energy",&tagger.mip_energy);
  reader.AddVariable("mip_n_end_reduction",&tagger.mip_n_end_reduction);
  reader.AddVariable("mip_n_first_mip",&tagger.mip_n_first_mip);
  reader.AddVariable("mip_n_first_non_mip",&tagger.mip_n_first_non_mip);
  reader.AddVariable("mip_n_first_non_mip_1",&tagger.mip_n_first_non_mip_1);
  reader.AddVariable("mip_n_first_non_mip_2",&tagger.mip_n_first_non_mip_2);
  reader.AddVariable("mip_vec_dQ_dx_0",&tagger.mip_vec_dQ_dx_0);
  reader.AddVariable("mip_vec_dQ_dx_1",&tagger.mip_vec_dQ_dx_1);
  reader.AddVariable("mip_max_dQ_dx_sample",&tagger.mip_max_dQ_dx_sample);
  reader.AddVariable("mip_n_below_threshold",&tagger.mip_n_below_threshold);
  reader.AddVariable("mip_n_below_zero",&tagger.mip_n_below_zero);
  reader.AddVariable("mip_n_lowest",&tagger.mip_n_lowest);
  reader.AddVariable("mip_n_highest",&tagger.mip_n_highest);
  reader.AddVariable("mip_lowest_dQ_dx",&tagger.mip_lowest_dQ_dx);
  reader.AddVariable("mip_highest_dQ_dx",&tagger.mip_highest_dQ_dx);
  reader.AddVariable("mip_medium_dQ_dx",&tagger.mip_medium_dQ_dx);
  reader.AddVariable("mip_stem_length",&tagger.mip_stem_length);
  reader.AddVariable("mip_length_main",&tagger.mip_length_main);
  reader.AddVariable("mip_length_total",&tagger.mip_length_total);
  reader.AddVariable("mip_angle_beam",&tagger.mip_angle_beam);
  reader.AddVariable("mip_iso_angle",&tagger.mip_iso_angle);
  reader.AddVariable("mip_n_vertex",&tagger.mip_n_vertex);
  reader.AddVariable("mip_n_good_tracks",&tagger.mip_n_good_tracks);
  reader.AddVariable("mip_E_indirect_max_energy",&tagger.mip_E_indirect_max_energy);
  reader.AddVariable("mip_flag_all_above",&tagger.mip_flag_all_above);
  reader.AddVariable("mip_min_dQ_dx_5",&tagger.mip_min_dQ_dx_5);
  reader.AddVariable("mip_n_other_vertex",&tagger.mip_n_other_vertex);
  reader.AddVariable("mip_n_stem_size",&tagger.mip_n_stem_size);
  reader.AddVariable("mip_flag_stem_trajectory",&tagger.mip_flag_stem_trajectory);
  reader.AddVariable("mip_min_dis",&tagger.mip_min_dis);
  reader.AddVariable("mip_vec_dQ_dx_2",&tagger.mip_vec_dQ_dx_2);
  reader.AddVariable("mip_vec_dQ_dx_3",&tagger.mip_vec_dQ_dx_3);
  reader.AddVariable("mip_vec_dQ_dx_4",&tagger.mip_vec_dQ_dx_4);
  reader.AddVariable("mip_vec_dQ_dx_5",&tagger.mip_vec_dQ_dx_5);
  reader.AddVariable("mip_vec_dQ_dx_6",&tagger.mip_vec_dQ_dx_6);
  reader.AddVariable("mip_vec_dQ_dx_7",&tagger.mip_vec_dQ_dx_7);
  reader.AddVariable("mip_vec_dQ_dx_8",&tagger.mip_vec_dQ_dx_8);
  reader.AddVariable("mip_vec_dQ_dx_9",&tagger.mip_vec_dQ_dx_9);
  reader.AddVariable("mip_vec_dQ_dx_10",&tagger.mip_vec_dQ_dx_10);
  reader.AddVariable("mip_vec_dQ_dx_11",&tagger.mip_vec_dQ_dx_11);
  reader.AddVariable("mip_vec_dQ_dx_12",&tagger.mip_vec_dQ_dx_12);
  reader.AddVariable("mip_vec_dQ_dx_13",&tagger.mip_vec_dQ_dx_13);
  reader.AddVariable("mip_vec_dQ_dx_14",&tagger.mip_vec_dQ_dx_14);
  reader.AddVariable("mip_vec_dQ_dx_15",&tagger.mip_vec_dQ_dx_15);
  reader.AddVariable("mip_vec_dQ_dx_16",&tagger.mip_vec_dQ_dx_16);
  reader.AddVariable("mip_vec_dQ_dx_17",&tagger.mip_vec_dQ_dx_17);
  reader.AddVariable("mip_vec_dQ_dx_18",&tagger.mip_vec_dQ_dx_18);
  reader.AddVariable("mip_vec_dQ_dx_19",&tagger.mip_vec_dQ_dx_19);
  reader.AddVariable("br3_3_score",&tagger.br3_3_score);
  reader.AddVariable("br3_5_score",&tagger.br3_5_score);
  reader.AddVariable("br3_6_score",&tagger.br3_6_score);
  reader.AddVariable("pio_2_score",&tagger.pio_2_score);
  reader.AddVariable("stw_2_score",&tagger.stw_2_score);
  reader.AddVariable("stw_3_score",&tagger.stw_3_score);
  reader.AddVariable("stw_4_score",&tagger.stw_4_score);
  reader.AddVariable("sig_1_score",&tagger.sig_1_score);
  reader.AddVariable("sig_2_score",&tagger.sig_2_score);
  reader.AddVariable("lol_1_score",&tagger.lol_1_score);
  reader.AddVariable("lol_2_score",&tagger.lol_2_score);
  reader.AddVariable("tro_1_score",&tagger.tro_1_score);
  reader.AddVariable("tro_2_score",&tagger.tro_2_score);
  reader.AddVariable("tro_4_score",&tagger.tro_4_score);
  reader.AddVariable("tro_5_score",&tagger.tro_5_score);
  reader.AddVariable("br4_1_shower_main_length",&tagger.br4_1_shower_main_length);
  reader.AddVariable("br4_1_shower_total_length",&tagger.br4_1_shower_total_length);
  reader.AddVariable("br4_1_min_dis",&tagger.br4_1_min_dis);
  reader.AddVariable("br4_1_energy",&tagger.br4_1_energy);
  reader.AddVariable("br4_1_flag_avoid_muon_check",&tagger.br4_1_flag_avoid_muon_check);
  reader.AddVariable("br4_1_n_vtx_segs",&tagger.br4_1_n_vtx_segs);
  reader.AddVariable("br4_2_ratio_45",&tagger.br4_2_ratio_45);
  reader.AddVariable("br4_2_ratio_35",&tagger.br4_2_ratio_35);
  reader.AddVariable("br4_2_ratio_25",&tagger.br4_2_ratio_25);
  reader.AddVariable("br4_2_ratio_15",&tagger.br4_2_ratio_15);
  reader.AddVariable("br4_2_ratio1_45",&tagger.br4_2_ratio1_45);
  reader.AddVariable("br4_2_ratio1_35",&tagger.br4_2_ratio1_35);
  reader.AddVariable("br4_2_ratio1_25",&tagger.br4_2_ratio1_25);
  reader.AddVariable("br4_2_ratio1_15",&tagger.br4_2_ratio1_15);
  reader.AddVariable("br4_2_iso_angle",&tagger.br4_2_iso_angle);
  reader.AddVariable("br4_2_iso_angle1",&tagger.br4_2_iso_angle1);
  reader.AddVariable("br4_2_angle",&tagger.br4_2_angle);
  reader.AddVariable("tro_3_stem_length",&tagger.tro_3_stem_length);
  reader.AddVariable("tro_3_n_muon_segs",&tagger.tro_3_n_muon_segs);
  reader.AddVariable("br4_1_n_main_segs",&tagger.br4_1_n_main_segs); // naming issue

  reader.BookMVA( "MyBDT", "./weights/XGB_nue_seed2_0923.xml");
  //  reader.BookMVA( "MyBDT", "./weights/xgboost_set8seed7_kaicheng_0819.xml");

  TMVA::Reader reader_cosmict_10;
  float cosmict_10_vtx_z;
  float cosmict_10_flag_shower;
  float cosmict_10_flag_dir_weak;
  float cosmict_10_angle_beam;
  float cosmict_10_length;

  reader_cosmict_10.AddVariable("cosmict_10_vtx_z",&cosmict_10_vtx_z);
  reader_cosmict_10.AddVariable("cosmict_10_flag_shower",&cosmict_10_flag_shower);
  reader_cosmict_10.AddVariable("cosmict_10_flag_dir_weak",&cosmict_10_flag_dir_weak);
  reader_cosmict_10.AddVariable("cosmict_10_angle_beam",&cosmict_10_angle_beam);
  reader_cosmict_10.AddVariable("cosmict_10_length",&cosmict_10_length);

  reader_cosmict_10.BookMVA( "MyBDT", "weights/cos_tagger_10.weights.xml");


  TMVA::Reader reader_numu_1;

  float numu_cc_flag_1;
  float numu_cc_1_particle_type;
  float numu_cc_1_length;
  float numu_cc_1_medium_dQ_dx;
  float numu_cc_1_dQ_dx_cut;
  float numu_cc_1_direct_length;
  float numu_cc_1_n_daughter_tracks;
  float numu_cc_1_n_daughter_all;

  reader_numu_1.AddVariable("numu_cc_1_particle_type",&numu_cc_1_particle_type);
  reader_numu_1.AddVariable("numu_cc_1_length",&numu_cc_1_length);
  reader_numu_1.AddVariable("numu_cc_1_medium_dQ_dx",&numu_cc_1_medium_dQ_dx);
  reader_numu_1.AddVariable("numu_cc_1_dQ_dx_cut",&numu_cc_1_dQ_dx_cut);
  reader_numu_1.AddVariable("numu_cc_1_direct_length",&numu_cc_1_direct_length);
  reader_numu_1.AddVariable("numu_cc_1_n_daughter_tracks",&numu_cc_1_n_daughter_tracks);
  reader_numu_1.AddVariable("numu_cc_1_n_daughter_all",&numu_cc_1_n_daughter_all);

  reader_numu_1.BookMVA( "MyBDT", "weights/numu_tagger1.weights.xml");


  TMVA::Reader reader_numu_2;
  float numu_cc_2_length;
  float numu_cc_2_total_length;
  float numu_cc_2_n_daughter_tracks;
  float numu_cc_2_n_daughter_all;

  reader_numu_2.AddVariable("numu_cc_2_length",&numu_cc_2_length);
  reader_numu_2.AddVariable("numu_cc_2_total_length",&numu_cc_2_total_length);
  reader_numu_2.AddVariable("numu_cc_2_n_daughter_tracks",&numu_cc_2_n_daughter_tracks);
  reader_numu_2.AddVariable("numu_cc_2_n_daughter_all",&numu_cc_2_n_daughter_all);

  reader_numu_2.BookMVA( "MyBDT", "weights/numu_tagger2.weights.xml");


  TMVA::Reader reader_numu;

  reader_numu.AddVariable("numu_cc_flag_3", &tagger.numu_cc_flag_3);
  reader_numu.AddVariable("numu_cc_3_particle_type", &tagger.numu_cc_3_particle_type);
  reader_numu.AddVariable("numu_cc_3_max_length", &tagger.numu_cc_3_max_length);
  reader_numu.AddVariable("numu_cc_3_track_length",&tagger.numu_cc_3_acc_track_length);
  reader_numu.AddVariable("numu_cc_3_max_length_all",&tagger.numu_cc_3_max_length_all);
  reader_numu.AddVariable("numu_cc_3_max_muon_length",&tagger.numu_cc_3_max_muon_length);
  reader_numu.AddVariable("numu_cc_3_n_daughter_tracks",&tagger.numu_cc_3_n_daughter_tracks);
  reader_numu.AddVariable("numu_cc_3_n_daughter_all",&tagger.numu_cc_3_n_daughter_all);
  reader_numu.AddVariable("cosmict_flag_2", &tagger.cosmict_flag_2);
  reader_numu.AddVariable("cosmict_2_filled", &tagger.cosmict_2_filled);
  reader_numu.AddVariable("cosmict_2_particle_type",&tagger.cosmict_2_particle_type);
  reader_numu.AddVariable("cosmict_2_n_muon_tracks",&tagger.cosmict_2_n_muon_tracks);
  reader_numu.AddVariable("cosmict_2_total_shower_length",&tagger.cosmict_2_total_shower_length);
  reader_numu.AddVariable("cosmict_2_flag_inside",&tagger.cosmict_2_flag_inside);
  reader_numu.AddVariable("cosmict_2_angle_beam",&tagger.cosmict_2_angle_beam);
  reader_numu.AddVariable("cosmict_2_flag_dir_weak", &tagger.cosmict_2_flag_dir_weak);
  reader_numu.AddVariable("cosmict_2_dQ_dx_end", &tagger.cosmict_2_dQ_dx_end);
  reader_numu.AddVariable("cosmict_2_dQ_dx_front", &tagger.cosmict_2_dQ_dx_front);
  reader_numu.AddVariable("cosmict_2_theta", &tagger.cosmict_2_theta);
  reader_numu.AddVariable("cosmict_2_phi", &tagger.cosmict_2_phi);
  reader_numu.AddVariable("cosmict_2_valid_tracks", &tagger.cosmict_2_valid_tracks);
  reader_numu.AddVariable("cosmict_flag_4", &tagger.cosmict_flag_4);
  reader_numu.AddVariable("cosmict_4_filled", &tagger.cosmict_4_filled);
  reader_numu.AddVariable("cosmict_4_flag_inside", &tagger.cosmict_4_flag_inside);
  reader_numu.AddVariable("cosmict_4_angle_beam", &tagger.cosmict_4_angle_beam);
  reader_numu.AddVariable("cosmict_4_connected_showers", &tagger.cosmict_4_connected_showers);
  reader_numu.AddVariable("cosmict_flag_3", &tagger.cosmict_flag_3);
  reader_numu.AddVariable("cosmict_3_filled", &tagger.cosmict_3_filled);
  reader_numu.AddVariable("cosmict_3_flag_inside", &tagger.cosmict_3_flag_inside);
  reader_numu.AddVariable("cosmict_3_angle_beam", &tagger.cosmict_3_angle_beam);
  reader_numu.AddVariable("cosmict_3_flag_dir_weak", &tagger.cosmict_3_flag_dir_weak);
  reader_numu.AddVariable("cosmict_3_dQ_dx_end", &tagger.cosmict_3_dQ_dx_end);
  reader_numu.AddVariable("cosmict_3_dQ_dx_front", &tagger.cosmict_3_dQ_dx_front);
  reader_numu.AddVariable("cosmict_3_theta", &tagger.cosmict_3_theta);
  reader_numu.AddVariable("cosmict_3_phi", &tagger.cosmict_3_phi);
  reader_numu.AddVariable("cosmict_3_valid_tracks", &tagger.cosmict_3_valid_tracks);
  reader_numu.AddVariable("cosmict_flag_5", &tagger.cosmict_flag_5);
  reader_numu.AddVariable("cosmict_5_filled", &tagger.cosmict_5_filled);
  reader_numu.AddVariable("cosmict_5_flag_inside", &tagger.cosmict_5_flag_inside);
  reader_numu.AddVariable("cosmict_5_angle_beam", &tagger.cosmict_5_angle_beam);
  reader_numu.AddVariable("cosmict_5_connected_showers", &tagger.cosmict_5_connected_showers);
  reader_numu.AddVariable("cosmict_flag_6", &tagger.cosmict_flag_6);
  reader_numu.AddVariable("cosmict_6_filled", &tagger.cosmict_6_filled);
  reader_numu.AddVariable("cosmict_6_flag_dir_weak", &tagger.cosmict_6_flag_dir_weak);
  reader_numu.AddVariable("cosmict_6_flag_inside", &tagger.cosmict_6_flag_inside);
  reader_numu.AddVariable("cosmict_6_angle", &tagger.cosmict_6_angle);
  reader_numu.AddVariable("cosmict_flag_7", &tagger.cosmict_flag_7);
  reader_numu.AddVariable("cosmict_7_filled", &tagger.cosmict_7_filled);
  reader_numu.AddVariable("cosmict_7_flag_sec", &tagger.cosmict_7_flag_sec);
  reader_numu.AddVariable("cosmict_7_n_muon_tracks", &tagger.cosmict_7_n_muon_tracks);
  reader_numu.AddVariable("cosmict_7_total_shower_length", &tagger.cosmict_7_total_shower_length);
  reader_numu.AddVariable("cosmict_7_flag_inside", &tagger.cosmict_7_flag_inside);
  reader_numu.AddVariable("cosmict_7_angle_beam", &tagger.cosmict_7_angle_beam);
  reader_numu.AddVariable("cosmict_7_flag_dir_weak", &tagger.cosmict_7_flag_dir_weak);
  reader_numu.AddVariable("cosmict_7_dQ_dx_end", &tagger.cosmict_7_dQ_dx_end);
  reader_numu.AddVariable("cosmict_7_dQ_dx_front", &tagger.cosmict_7_dQ_dx_front);
  reader_numu.AddVariable("cosmict_7_theta", &tagger.cosmict_7_theta);
  reader_numu.AddVariable("cosmict_7_phi", &tagger.cosmict_7_phi);
  reader_numu.AddVariable("cosmict_flag_8", &tagger.cosmict_flag_8);
  reader_numu.AddVariable("cosmict_8_filled", &tagger.cosmict_8_filled);
  reader_numu.AddVariable("cosmict_8_flag_out", &tagger.cosmict_8_flag_out);
  reader_numu.AddVariable("cosmict_8_muon_length", &tagger.cosmict_8_muon_length);
  reader_numu.AddVariable("cosmict_8_acc_length", &tagger.cosmict_8_acc_length);
  reader_numu.AddVariable("cosmict_flag_9", &tagger.cosmict_flag_9);
  reader_numu.AddVariable("cosmic_flag", &tagger.cosmic_flag);
  reader_numu.AddVariable("cosmic_filled", &tagger.cosmic_filled);
  reader_numu.AddVariable("cosmict_flag", &tagger.cosmict_flag);
  reader_numu.AddVariable("numu_cc_flag", &tagger.numu_cc_flag);
  reader_numu.AddVariable("cosmict_flag_1", &tagger.cosmict_flag_1);
  reader_numu.AddVariable("kine_reco_Enu",&tagger.kine_reco_Enu);
  reader_numu.AddVariable("match_isFC",&tagger.match_isFC);
  reader_numu.AddVariable("cosmict_10_score", &tagger.cosmict_10_score);
  reader_numu.AddVariable("numu_1_score", &tagger.numu_1_score);
  reader_numu.AddVariable("numu_2_score", &tagger.numu_2_score);

  reader_numu.BookMVA( "MyBDT", "weights/numu_scalars_scores_0923.xml");

  // NC delta BDT ... by Lee @ Yale
  TMVA::Reader reader_nc_delta;

  reader_nc_delta.AddVariable("numu_cc_flag_3",&tagger.numu_cc_flag_3);
  reader_nc_delta.AddVariable("numu_cc_3_particle_type",&tagger.numu_cc_3_particle_type);
  reader_nc_delta.AddVariable("numu_cc_3_max_length",&tagger.numu_cc_3_max_length);
  reader_nc_delta.AddVariable("numu_cc_3_track_length",&tagger.numu_cc_3_acc_track_length);
  reader_nc_delta.AddVariable("numu_cc_3_max_length_all",&tagger.numu_cc_3_max_length_all);
  reader_nc_delta.AddVariable("numu_cc_3_max_muon_length",&tagger.numu_cc_3_max_muon_length);
  reader_nc_delta.AddVariable("numu_cc_3_n_daughter_tracks",&tagger.numu_cc_3_n_daughter_tracks);
  reader_nc_delta.AddVariable("numu_cc_3_n_daughter_all",&tagger.numu_cc_3_n_daughter_all);
  reader_nc_delta.AddVariable("cosmict_flag_2",&tagger.cosmict_flag_2);
  reader_nc_delta.AddVariable("cosmict_2_filled",&tagger.cosmict_2_filled);
  reader_nc_delta.AddVariable("cosmict_2_particle_type",&tagger.cosmict_2_particle_type);
  reader_nc_delta.AddVariable("cosmict_2_n_muon_tracks",&tagger.cosmict_2_n_muon_tracks);
  reader_nc_delta.AddVariable("cosmict_2_total_shower_length",&tagger.cosmict_2_total_shower_length);
  reader_nc_delta.AddVariable("cosmict_2_flag_inside",&tagger.cosmict_2_flag_inside);
  reader_nc_delta.AddVariable("cosmict_2_angle_beam",&tagger.cosmict_2_angle_beam);
  reader_nc_delta.AddVariable("cosmict_2_flag_dir_weak",&tagger.cosmict_2_flag_dir_weak);
  reader_nc_delta.AddVariable("cosmict_2_dQ_dx_end",&tagger.cosmict_2_dQ_dx_end);
  reader_nc_delta.AddVariable("cosmict_2_dQ_dx_front",&tagger.cosmict_2_dQ_dx_front);
  reader_nc_delta.AddVariable("cosmict_2_theta",&tagger.cosmict_2_theta);
  reader_nc_delta.AddVariable("cosmict_2_phi",&tagger.cosmict_2_phi);
  reader_nc_delta.AddVariable("cosmict_2_valid_tracks",&tagger.cosmict_2_valid_tracks);
  reader_nc_delta.AddVariable("cosmict_flag_4",&tagger.cosmict_flag_4);
  reader_nc_delta.AddVariable("cosmict_4_filled",&tagger.cosmict_4_filled);
  reader_nc_delta.AddVariable("cosmict_4_flag_inside",&tagger.cosmict_4_flag_inside);
  reader_nc_delta.AddVariable("cosmict_4_angle_beam",&tagger.cosmict_4_angle_beam);
  reader_nc_delta.AddVariable("cosmict_4_connected_showers",&tagger.cosmict_4_connected_showers);
  reader_nc_delta.AddVariable("cosmict_flag_3",&tagger.cosmict_flag_3);
  reader_nc_delta.AddVariable("cosmict_3_filled",&tagger.cosmict_3_filled);
  reader_nc_delta.AddVariable("cosmict_3_flag_inside",&tagger.cosmict_3_flag_inside);
  reader_nc_delta.AddVariable("cosmict_3_angle_beam",&tagger.cosmict_3_angle_beam);
  reader_nc_delta.AddVariable("cosmict_3_flag_dir_weak",&tagger.cosmict_3_flag_dir_weak);
  reader_nc_delta.AddVariable("cosmict_3_dQ_dx_end",&tagger.cosmict_3_dQ_dx_end);
  reader_nc_delta.AddVariable("cosmict_3_dQ_dx_front",&tagger.cosmict_3_dQ_dx_front);
  reader_nc_delta.AddVariable("cosmict_3_theta",&tagger.cosmict_3_theta);
  reader_nc_delta.AddVariable("cosmict_3_phi",&tagger.cosmict_3_phi);
  reader_nc_delta.AddVariable("cosmict_3_valid_tracks",&tagger.cosmict_3_valid_tracks);
  reader_nc_delta.AddVariable("cosmict_flag_5",&tagger.cosmict_flag_5);
  reader_nc_delta.AddVariable("cosmict_5_filled",&tagger.cosmict_5_filled);
  reader_nc_delta.AddVariable("cosmict_5_flag_inside",&tagger.cosmict_5_flag_inside);
  reader_nc_delta.AddVariable("cosmict_5_angle_beam",&tagger.cosmict_5_angle_beam);
  reader_nc_delta.AddVariable("cosmict_5_connected_showers",&tagger.cosmict_5_connected_showers);
  reader_nc_delta.AddVariable("cosmict_flag_6",&tagger.cosmict_flag_6);
  reader_nc_delta.AddVariable("cosmict_6_filled",&tagger.cosmict_6_filled);
  reader_nc_delta.AddVariable("cosmict_6_flag_dir_weak",&tagger.cosmict_6_flag_dir_weak);
  reader_nc_delta.AddVariable("cosmict_6_flag_inside",&tagger.cosmict_6_flag_inside);
  reader_nc_delta.AddVariable("cosmict_6_angle",&tagger.cosmict_6_angle);
  reader_nc_delta.AddVariable("cosmict_flag_7",&tagger.cosmict_flag_7);
  reader_nc_delta.AddVariable("cosmict_7_filled",&tagger.cosmict_7_filled);
  reader_nc_delta.AddVariable("cosmict_7_flag_sec",&tagger.cosmict_7_flag_sec);
  reader_nc_delta.AddVariable("cosmict_7_n_muon_tracks",&tagger.cosmict_7_n_muon_tracks);
  reader_nc_delta.AddVariable("cosmict_7_total_shower_length",&tagger.cosmict_7_total_shower_length);
  reader_nc_delta.AddVariable("cosmict_7_flag_inside",&tagger.cosmict_7_flag_inside);
  reader_nc_delta.AddVariable("cosmict_7_angle_beam",&tagger.cosmict_7_angle_beam);
  reader_nc_delta.AddVariable("cosmict_7_flag_dir_weak",&tagger.cosmict_7_flag_dir_weak);
  reader_nc_delta.AddVariable("cosmict_7_dQ_dx_end",&tagger.cosmict_7_dQ_dx_end);
  reader_nc_delta.AddVariable("cosmict_7_dQ_dx_front",&tagger.cosmict_7_dQ_dx_front);
  reader_nc_delta.AddVariable("cosmict_7_theta",&tagger.cosmict_7_theta);
  reader_nc_delta.AddVariable("cosmict_7_phi",&tagger.cosmict_7_phi);
  reader_nc_delta.AddVariable("cosmict_flag_8",&tagger.cosmict_flag_8);
  reader_nc_delta.AddVariable("cosmict_8_filled",&tagger.cosmict_8_filled);
  reader_nc_delta.AddVariable("cosmict_8_flag_out",&tagger.cosmict_8_flag_out);
  reader_nc_delta.AddVariable("cosmict_8_muon_length",&tagger.cosmict_8_muon_length);
  reader_nc_delta.AddVariable("cosmict_8_acc_length",&tagger.cosmict_8_acc_length);
  reader_nc_delta.AddVariable("cosmict_flag_9",&tagger.cosmict_flag_9);
  reader_nc_delta.AddVariable("cosmic_flag",&tagger.cosmic_flag);
  reader_nc_delta.AddVariable("cosmic_filled",&tagger.cosmic_filled);
  reader_nc_delta.AddVariable("cosmict_flag",&tagger.cosmict_flag);
  reader_nc_delta.AddVariable("numu_cc_flag",&tagger.numu_cc_flag);
  reader_nc_delta.AddVariable("cosmict_flag_1",&tagger.cosmict_flag_1);
  reader_nc_delta.AddVariable("cme_mu_energy",&tagger.cme_mu_energy);
  reader_nc_delta.AddVariable("cme_energy",&tagger.cme_energy);
  reader_nc_delta.AddVariable("cme_mu_length",&tagger.cme_mu_length);
  reader_nc_delta.AddVariable("cme_length",&tagger.cme_length);
  reader_nc_delta.AddVariable("cme_angle_beam",&tagger.cme_angle_beam);
  reader_nc_delta.AddVariable("anc_angle",&tagger.anc_angle);
  reader_nc_delta.AddVariable("anc_max_angle",&tagger.anc_max_angle);
  reader_nc_delta.AddVariable("anc_max_length",&tagger.anc_max_length);
  reader_nc_delta.AddVariable("anc_acc_forward_length",&tagger.anc_acc_forward_length);
  reader_nc_delta.AddVariable("anc_acc_backward_length",&tagger.anc_acc_backward_length);
  reader_nc_delta.AddVariable("anc_acc_forward_length1",&tagger.anc_acc_forward_length1);
  reader_nc_delta.AddVariable("anc_shower_main_length",&tagger.anc_shower_main_length);
  reader_nc_delta.AddVariable("anc_shower_total_length",&tagger.anc_shower_total_length);
  reader_nc_delta.AddVariable("anc_flag_main_outside",&tagger.anc_flag_main_outside);
  reader_nc_delta.AddVariable("gap_flag_prolong_u",&tagger.gap_flag_prolong_u);
  reader_nc_delta.AddVariable("gap_flag_prolong_v",&tagger.gap_flag_prolong_v);
  reader_nc_delta.AddVariable("gap_flag_prolong_w",&tagger.gap_flag_prolong_w);
  reader_nc_delta.AddVariable("gap_flag_parallel",&tagger.gap_flag_parallel);
  reader_nc_delta.AddVariable("gap_n_points",&tagger.gap_n_points);
  reader_nc_delta.AddVariable("gap_n_bad",&tagger.gap_n_bad);
  reader_nc_delta.AddVariable("gap_energy",&tagger.gap_energy);
  reader_nc_delta.AddVariable("gap_num_valid_tracks",&tagger.gap_num_valid_tracks);
  reader_nc_delta.AddVariable("gap_flag_single_shower",&tagger.gap_flag_single_shower);
  reader_nc_delta.AddVariable("hol_1_n_valid_tracks",&tagger.hol_1_n_valid_tracks);
  reader_nc_delta.AddVariable("hol_1_min_angle",&tagger.hol_1_min_angle);
  reader_nc_delta.AddVariable("hol_1_energy",&tagger.hol_1_energy);
  reader_nc_delta.AddVariable("hol_1_flag_all_shower",&tagger.hol_1_flag_all_shower);
  reader_nc_delta.AddVariable("hol_1_min_length",&tagger.hol_1_min_length);
  reader_nc_delta.AddVariable("hol_2_min_angle",&tagger.hol_2_min_angle);
  reader_nc_delta.AddVariable("hol_2_medium_dQ_dx",&tagger.hol_2_medium_dQ_dx);
  reader_nc_delta.AddVariable("hol_2_ncount",&tagger.hol_2_ncount);
  reader_nc_delta.AddVariable("lol_3_angle_beam",&tagger.lol_3_angle_beam);
  reader_nc_delta.AddVariable("lol_3_n_valid_tracks",&tagger.lol_3_n_valid_tracks);
  reader_nc_delta.AddVariable("lol_3_min_angle",&tagger.lol_3_min_angle);
  reader_nc_delta.AddVariable("lol_3_vtx_n_segs",&tagger.lol_3_vtx_n_segs);
  reader_nc_delta.AddVariable("lol_3_shower_main_length",&tagger.lol_3_shower_main_length);
  reader_nc_delta.AddVariable("lol_3_n_out",&tagger.lol_3_n_out);
  reader_nc_delta.AddVariable("lol_3_n_sum",&tagger.lol_3_n_sum);
  reader_nc_delta.AddVariable("mgo_energy",&tagger.mgo_energy);
  reader_nc_delta.AddVariable("mgo_max_energy",&tagger.mgo_max_energy);
  reader_nc_delta.AddVariable("mgo_total_energy",&tagger.mgo_total_energy);
  reader_nc_delta.AddVariable("mgo_n_showers",&tagger.mgo_n_showers);
  reader_nc_delta.AddVariable("mgo_max_energy_1",&tagger.mgo_max_energy_1);
  reader_nc_delta.AddVariable("mgo_max_energy_2",&tagger.mgo_max_energy_2);
  reader_nc_delta.AddVariable("mgo_total_other_energy",&tagger.mgo_total_other_energy);
  reader_nc_delta.AddVariable("mgo_n_total_showers",&tagger.mgo_n_total_showers);
  reader_nc_delta.AddVariable("mgo_total_other_energy_1",&tagger.mgo_total_other_energy_1);
  reader_nc_delta.AddVariable("mgt_flag_single_shower",&tagger.mgt_flag_single_shower);
  reader_nc_delta.AddVariable("mgt_max_energy",&tagger.mgt_max_energy);
  reader_nc_delta.AddVariable("mgt_total_other_energy",&tagger.mgt_total_other_energy);
  reader_nc_delta.AddVariable("mgt_max_energy_1",&tagger.mgt_max_energy_1);
  reader_nc_delta.AddVariable("mgt_e_indirect_max_energy",&tagger.mgt_e_indirect_max_energy);
  reader_nc_delta.AddVariable("mgt_e_direct_max_energy",&tagger.mgt_e_direct_max_energy);
  reader_nc_delta.AddVariable("mgt_n_direct_showers",&tagger.mgt_n_direct_showers);
  reader_nc_delta.AddVariable("mgt_e_direct_total_energy",&tagger.mgt_e_direct_total_energy);
  reader_nc_delta.AddVariable("mgt_flag_indirect_max_pio",&tagger.mgt_flag_indirect_max_pio);
  reader_nc_delta.AddVariable("mgt_e_indirect_total_energy",&tagger.mgt_e_indirect_total_energy);
  reader_nc_delta.AddVariable("mip_quality_energy",&tagger.mip_quality_energy);
  reader_nc_delta.AddVariable("mip_quality_overlap",&tagger.mip_quality_overlap);
  reader_nc_delta.AddVariable("mip_quality_n_showers",&tagger.mip_quality_n_showers);
  reader_nc_delta.AddVariable("mip_quality_n_tracks",&tagger.mip_quality_n_tracks);
  reader_nc_delta.AddVariable("mip_quality_flag_inside_pi0",&tagger.mip_quality_flag_inside_pi0);
  reader_nc_delta.AddVariable("mip_quality_n_pi0_showers",&tagger.mip_quality_n_pi0_showers);
  reader_nc_delta.AddVariable("mip_quality_shortest_length",&tagger.mip_quality_shortest_length);
  reader_nc_delta.AddVariable("mip_quality_acc_length",&tagger.mip_quality_acc_length);
  reader_nc_delta.AddVariable("mip_quality_shortest_angle",&tagger.mip_quality_shortest_angle);
  reader_nc_delta.AddVariable("mip_quality_flag_proton",&tagger.mip_quality_flag_proton);
  reader_nc_delta.AddVariable("br1_1_shower_type",&tagger.br1_1_shower_type);
  reader_nc_delta.AddVariable("br1_1_vtx_n_segs",&tagger.br1_1_vtx_n_segs);
  reader_nc_delta.AddVariable("br1_1_energy",&tagger.br1_1_energy);
  reader_nc_delta.AddVariable("br1_1_n_segs",&tagger.br1_1_n_segs);
  reader_nc_delta.AddVariable("br1_1_flag_sg_topology",&tagger.br1_1_flag_sg_topology);
  reader_nc_delta.AddVariable("br1_1_flag_sg_trajectory",&tagger.br1_1_flag_sg_trajectory);
  reader_nc_delta.AddVariable("br1_1_sg_length",&tagger.br1_1_sg_length);
  reader_nc_delta.AddVariable("br1_2_n_connected",&tagger.br1_2_n_connected);
  reader_nc_delta.AddVariable("br1_2_max_length",&tagger.br1_2_max_length);
  reader_nc_delta.AddVariable("br1_2_n_connected_1",&tagger.br1_2_n_connected_1);
  reader_nc_delta.AddVariable("br1_2_n_shower_segs",&tagger.br1_2_n_shower_segs);
  reader_nc_delta.AddVariable("br1_2_max_length_ratio",&tagger.br1_2_max_length_ratio);
  reader_nc_delta.AddVariable("br1_2_shower_length",&tagger.br1_2_shower_length);
  reader_nc_delta.AddVariable("br1_3_n_connected_p",&tagger.br1_3_n_connected_p);
  reader_nc_delta.AddVariable("br1_3_max_length_p",&tagger.br1_3_max_length_p);
  reader_nc_delta.AddVariable("br1_3_n_shower_main_segs",&tagger.br1_3_n_shower_main_segs);
  reader_nc_delta.AddVariable("br3_1_energy",&tagger.br3_1_energy);
  reader_nc_delta.AddVariable("br3_1_n_shower_segments",&tagger.br3_1_n_shower_segments);
  reader_nc_delta.AddVariable("br3_1_sg_flag_trajectory",&tagger.br3_1_sg_flag_trajectory);
  reader_nc_delta.AddVariable("br3_1_sg_direct_length",&tagger.br3_1_sg_direct_length);
  reader_nc_delta.AddVariable("br3_1_sg_length",&tagger.br3_1_sg_length);
  reader_nc_delta.AddVariable("br3_1_total_main_length",&tagger.br3_1_total_main_length);
  reader_nc_delta.AddVariable("br3_1_total_length",&tagger.br3_1_total_length);
  reader_nc_delta.AddVariable("br3_1_iso_angle",&tagger.br3_1_iso_angle);
  reader_nc_delta.AddVariable("br3_1_sg_flag_topology",&tagger.br3_1_sg_flag_topology);
  reader_nc_delta.AddVariable("br3_2_n_ele",&tagger.br3_2_n_ele);
  reader_nc_delta.AddVariable("br3_2_n_other",&tagger.br3_2_n_other);
  reader_nc_delta.AddVariable("br3_2_other_fid",&tagger.br3_2_other_fid);
  reader_nc_delta.AddVariable("br3_4_acc_length",&tagger.br3_4_acc_length);
  reader_nc_delta.AddVariable("br3_4_total_length",&tagger.br3_4_total_length);
  reader_nc_delta.AddVariable("br3_7_min_angle",&tagger.br3_7_min_angle);
  reader_nc_delta.AddVariable("br3_8_max_dQ_dx",&tagger.br3_8_max_dQ_dx);
  reader_nc_delta.AddVariable("br3_8_n_main_segs",&tagger.br3_8_n_main_segs);
  reader_nc_delta.AddVariable("br4_1_shower_main_length",&tagger.br4_1_shower_main_length);
  reader_nc_delta.AddVariable("br4_1_shower_total_length",&tagger.br4_1_shower_total_length);
  reader_nc_delta.AddVariable("br4_1_min_dis",&tagger.br4_1_min_dis);
  reader_nc_delta.AddVariable("br4_1_energy",&tagger.br4_1_energy);
  reader_nc_delta.AddVariable("br4_1_flag_avoid_muon_check",&tagger.br4_1_flag_avoid_muon_check);
  reader_nc_delta.AddVariable("br4_1_n_vtx_segs",&tagger.br4_1_n_vtx_segs);
  reader_nc_delta.AddVariable("br4_1_n_main_segs",&tagger.br4_1_n_main_segs);
  reader_nc_delta.AddVariable("br4_2_ratio_45",&tagger.br4_2_ratio_45);
  reader_nc_delta.AddVariable("br4_2_ratio_35",&tagger.br4_2_ratio_35);
  reader_nc_delta.AddVariable("br4_2_ratio_25",&tagger.br4_2_ratio_25);
  reader_nc_delta.AddVariable("br4_2_ratio_15",&tagger.br4_2_ratio_15);
  reader_nc_delta.AddVariable("br4_2_ratio1_45",&tagger.br4_2_ratio1_45);
  reader_nc_delta.AddVariable("br4_2_ratio1_35",&tagger.br4_2_ratio1_35);
  reader_nc_delta.AddVariable("br4_2_ratio1_25",&tagger.br4_2_ratio1_25);
  reader_nc_delta.AddVariable("br4_2_ratio1_15",&tagger.br4_2_ratio1_15);
  reader_nc_delta.AddVariable("br4_2_iso_angle",&tagger.br4_2_iso_angle);
  reader_nc_delta.AddVariable("br4_2_iso_angle1",&tagger.br4_2_iso_angle1);
  reader_nc_delta.AddVariable("br4_2_angle",&tagger.br4_2_angle);
  reader_nc_delta.AddVariable("tro_3_stem_length",&tagger.tro_3_stem_length);
  reader_nc_delta.AddVariable("tro_3_n_muon_segs",&tagger.tro_3_n_muon_segs);
  reader_nc_delta.AddVariable("stem_dir_flag_single_shower",&tagger.stem_dir_flag_single_shower);
  reader_nc_delta.AddVariable("stem_dir_angle",&tagger.stem_dir_angle);
  reader_nc_delta.AddVariable("stem_dir_energy",&tagger.stem_dir_energy);
  reader_nc_delta.AddVariable("stem_dir_angle1",&tagger.stem_dir_angle1);
  reader_nc_delta.AddVariable("stem_dir_angle2",&tagger.stem_dir_angle2);
  reader_nc_delta.AddVariable("stem_dir_angle3",&tagger.stem_dir_angle3);
  reader_nc_delta.AddVariable("stem_dir_ratio",&tagger.stem_dir_ratio);
  reader_nc_delta.AddVariable("br2_num_valid_tracks",&tagger.br2_num_valid_tracks);
  reader_nc_delta.AddVariable("br2_n_shower_main_segs",&tagger.br2_n_shower_main_segs);
  reader_nc_delta.AddVariable("br2_max_angle",&tagger.br2_max_angle);
  reader_nc_delta.AddVariable("br2_sg_length",&tagger.br2_sg_length);
  reader_nc_delta.AddVariable("br2_flag_sg_trajectory",&tagger.br2_flag_sg_trajectory);
  reader_nc_delta.AddVariable("stem_len_energy",&tagger.stem_len_energy);
  reader_nc_delta.AddVariable("stem_len_length",&tagger.stem_len_length);
  reader_nc_delta.AddVariable("stem_len_flag_avoid_muon_check",&tagger.stem_len_flag_avoid_muon_check);
  reader_nc_delta.AddVariable("stem_len_num_daughters",&tagger.stem_len_num_daughters);
  reader_nc_delta.AddVariable("stem_len_daughter_length",&tagger.stem_len_daughter_length);
  reader_nc_delta.AddVariable("brm_n_mu_segs",&tagger.brm_n_mu_segs);
  reader_nc_delta.AddVariable("brm_Ep",&tagger.brm_Ep);
  reader_nc_delta.AddVariable("brm_acc_length",&tagger.brm_acc_length);
  reader_nc_delta.AddVariable("brm_shower_total_length",&tagger.brm_shower_total_length);
  reader_nc_delta.AddVariable("brm_connected_length",&tagger.brm_connected_length);
  reader_nc_delta.AddVariable("brm_n_size",&tagger.brm_n_size);
  reader_nc_delta.AddVariable("brm_acc_direct_length",&tagger.brm_acc_direct_length);
  reader_nc_delta.AddVariable("brm_n_shower_main_segs",&tagger.brm_n_shower_main_segs);
  reader_nc_delta.AddVariable("brm_n_mu_main",&tagger.brm_n_mu_main);
  reader_nc_delta.AddVariable("lem_shower_main_length",&tagger.lem_shower_main_length);
  reader_nc_delta.AddVariable("lem_n_3seg",&tagger.lem_n_3seg);
  reader_nc_delta.AddVariable("lem_e_charge",&tagger.lem_e_charge);
  reader_nc_delta.AddVariable("lem_e_dQdx",&tagger.lem_e_dQdx);
  reader_nc_delta.AddVariable("lem_shower_num_main_segs",&tagger.lem_shower_num_main_segs);
  reader_nc_delta.AddVariable("stw_1_energy",&tagger.stw_1_energy);
  reader_nc_delta.AddVariable("stw_1_dis",&tagger.stw_1_dis);
  reader_nc_delta.AddVariable("stw_1_dQ_dx",&tagger.stw_1_dQ_dx);
  reader_nc_delta.AddVariable("stw_1_flag_single_shower",&tagger.stw_1_flag_single_shower);
  reader_nc_delta.AddVariable("stw_1_n_pi0",&tagger.stw_1_n_pi0);
  reader_nc_delta.AddVariable("stw_1_num_valid_tracks",&tagger.stw_1_num_valid_tracks);
  reader_nc_delta.AddVariable("spt_shower_main_length",&tagger.spt_shower_main_length);
  reader_nc_delta.AddVariable("spt_shower_total_length",&tagger.spt_shower_total_length);
  reader_nc_delta.AddVariable("spt_angle_beam",&tagger.spt_angle_beam);
  reader_nc_delta.AddVariable("spt_angle_vertical",&tagger.spt_angle_vertical);
  reader_nc_delta.AddVariable("spt_max_dQ_dx",&tagger.spt_max_dQ_dx);
  reader_nc_delta.AddVariable("spt_angle_beam_1",&tagger.spt_angle_beam_1);
  reader_nc_delta.AddVariable("spt_angle_drift",&tagger.spt_angle_drift);
  reader_nc_delta.AddVariable("spt_angle_drift_1",&tagger.spt_angle_drift_1);
  reader_nc_delta.AddVariable("spt_num_valid_tracks",&tagger.spt_num_valid_tracks);
  reader_nc_delta.AddVariable("spt_n_vtx_segs",&tagger.spt_n_vtx_segs);
  reader_nc_delta.AddVariable("spt_max_length",&tagger.spt_max_length);
  reader_nc_delta.AddVariable("mip_energy",&tagger.mip_energy);
  reader_nc_delta.AddVariable("mip_n_end_reduction",&tagger.mip_n_end_reduction);
  reader_nc_delta.AddVariable("mip_n_first_mip",&tagger.mip_n_first_mip);
  reader_nc_delta.AddVariable("mip_n_first_non_mip",&tagger.mip_n_first_non_mip);
  reader_nc_delta.AddVariable("mip_n_first_non_mip_1",&tagger.mip_n_first_non_mip_1);
  reader_nc_delta.AddVariable("mip_n_first_non_mip_2",&tagger.mip_n_first_non_mip_2);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_0",&tagger.mip_vec_dQ_dx_0);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_1",&tagger.mip_vec_dQ_dx_1);
  reader_nc_delta.AddVariable("mip_max_dQ_dx_sample",&tagger.mip_max_dQ_dx_sample);
  reader_nc_delta.AddVariable("mip_n_below_threshold",&tagger.mip_n_below_threshold);
  reader_nc_delta.AddVariable("mip_n_below_zero",&tagger.mip_n_below_zero);
  reader_nc_delta.AddVariable("mip_n_lowest",&tagger.mip_n_lowest);
  reader_nc_delta.AddVariable("mip_n_highest",&tagger.mip_n_highest);
  reader_nc_delta.AddVariable("mip_lowest_dQ_dx",&tagger.mip_lowest_dQ_dx);
  reader_nc_delta.AddVariable("mip_highest_dQ_dx",&tagger.mip_highest_dQ_dx);
  reader_nc_delta.AddVariable("mip_medium_dQ_dx",&tagger.mip_medium_dQ_dx);
  reader_nc_delta.AddVariable("mip_stem_length",&tagger.mip_stem_length);
  reader_nc_delta.AddVariable("mip_length_main",&tagger.mip_length_main);
  reader_nc_delta.AddVariable("mip_length_total",&tagger.mip_length_total);
  reader_nc_delta.AddVariable("mip_angle_beam",&tagger.mip_angle_beam);
  reader_nc_delta.AddVariable("mip_iso_angle",&tagger.mip_iso_angle);
  reader_nc_delta.AddVariable("mip_n_vertex",&tagger.mip_n_vertex);
  reader_nc_delta.AddVariable("mip_n_good_tracks",&tagger.mip_n_good_tracks);
  reader_nc_delta.AddVariable("mip_E_indirect_max_energy",&tagger.mip_E_indirect_max_energy);
  reader_nc_delta.AddVariable("mip_flag_all_above",&tagger.mip_flag_all_above);
  reader_nc_delta.AddVariable("mip_min_dQ_dx_5",&tagger.mip_min_dQ_dx_5);
  reader_nc_delta.AddVariable("mip_n_other_vertex",&tagger.mip_n_other_vertex);
  reader_nc_delta.AddVariable("mip_n_stem_size",&tagger.mip_n_stem_size);
  reader_nc_delta.AddVariable("mip_flag_stem_trajectory",&tagger.mip_flag_stem_trajectory);
  reader_nc_delta.AddVariable("mip_min_dis",&tagger.mip_min_dis);
  reader_nc_delta.AddVariable("vis_1_n_vtx_segs",&tagger.vis_1_n_vtx_segs);
  reader_nc_delta.AddVariable("vis_1_energy",&tagger.vis_1_energy);
  reader_nc_delta.AddVariable("vis_1_num_good_tracks",&tagger.vis_1_num_good_tracks);
  reader_nc_delta.AddVariable("vis_1_max_angle",&tagger.vis_1_max_angle);
  reader_nc_delta.AddVariable("vis_1_max_shower_angle",&tagger.vis_1_max_shower_angle);
  reader_nc_delta.AddVariable("vis_1_tmp_length1",&tagger.vis_1_tmp_length1);
  reader_nc_delta.AddVariable("vis_1_tmp_length2",&tagger.vis_1_tmp_length2);
  reader_nc_delta.AddVariable("vis_2_n_vtx_segs",&tagger.vis_2_n_vtx_segs);
  reader_nc_delta.AddVariable("vis_2_min_angle",&tagger.vis_2_min_angle);
  reader_nc_delta.AddVariable("vis_2_min_weak_track",&tagger.vis_2_min_weak_track);
  reader_nc_delta.AddVariable("vis_2_angle_beam",&tagger.vis_2_angle_beam);
  reader_nc_delta.AddVariable("vis_2_min_angle1",&tagger.vis_2_min_angle1);
  reader_nc_delta.AddVariable("vis_2_iso_angle1",&tagger.vis_2_iso_angle1);
  reader_nc_delta.AddVariable("vis_2_min_medium_dQ_dx",&tagger.vis_2_min_medium_dQ_dx);
  reader_nc_delta.AddVariable("vis_2_min_length",&tagger.vis_2_min_length);
  reader_nc_delta.AddVariable("vis_2_sg_length",&tagger.vis_2_sg_length);
  reader_nc_delta.AddVariable("vis_2_max_angle",&tagger.vis_2_max_angle);
  reader_nc_delta.AddVariable("vis_2_max_weak_track",&tagger.vis_2_max_weak_track);
  reader_nc_delta.AddVariable("pio_1_mass",&tagger.pio_1_mass);
  reader_nc_delta.AddVariable("pio_1_pio_type",&tagger.pio_1_pio_type);
  reader_nc_delta.AddVariable("pio_1_energy_1",&tagger.pio_1_energy_1);
  reader_nc_delta.AddVariable("pio_1_energy_2",&tagger.pio_1_energy_2);
  reader_nc_delta.AddVariable("pio_1_dis_1",&tagger.pio_1_dis_1);
  reader_nc_delta.AddVariable("pio_1_dis_2",&tagger.pio_1_dis_2);
  reader_nc_delta.AddVariable("pio_mip_id",&tagger.pio_mip_id);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_2",&tagger.mip_vec_dQ_dx_2);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_3",&tagger.mip_vec_dQ_dx_3);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_4",&tagger.mip_vec_dQ_dx_4);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_5",&tagger.mip_vec_dQ_dx_5);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_6",&tagger.mip_vec_dQ_dx_6);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_7",&tagger.mip_vec_dQ_dx_7);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_8",&tagger.mip_vec_dQ_dx_8);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_9",&tagger.mip_vec_dQ_dx_9);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_10",&tagger.mip_vec_dQ_dx_10);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_11",&tagger.mip_vec_dQ_dx_11);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_12",&tagger.mip_vec_dQ_dx_12);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_13",&tagger.mip_vec_dQ_dx_13);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_14",&tagger.mip_vec_dQ_dx_14);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_15",&tagger.mip_vec_dQ_dx_15);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_16",&tagger.mip_vec_dQ_dx_16);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_17",&tagger.mip_vec_dQ_dx_17);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_18",&tagger.mip_vec_dQ_dx_18);
  reader_nc_delta.AddVariable("mip_vec_dQ_dx_19",&tagger.mip_vec_dQ_dx_19);
  reader_nc_delta.AddVariable("cosmict_10_score",&tagger.cosmict_10_score);
  reader_nc_delta.AddVariable("numu_1_score",&tagger.numu_1_score);
  reader_nc_delta.AddVariable("numu_2_score",&tagger.numu_2_score);
  reader_nc_delta.AddVariable("tro_5_score",&tagger.tro_5_score);
  reader_nc_delta.AddVariable("tro_4_score",&tagger.tro_4_score);
  reader_nc_delta.AddVariable("tro_2_score",&tagger.tro_2_score);
  reader_nc_delta.AddVariable("tro_1_score",&tagger.tro_1_score);
  reader_nc_delta.AddVariable("stw_4_score",&tagger.stw_4_score);
  reader_nc_delta.AddVariable("stw_3_score",&tagger.stw_3_score);
  reader_nc_delta.AddVariable("stw_2_score",&tagger.stw_2_score);
  reader_nc_delta.AddVariable("sig_2_score",&tagger.sig_2_score);
  reader_nc_delta.AddVariable("sig_1_score",&tagger.sig_1_score);
  reader_nc_delta.AddVariable("pio_2_score",&tagger.pio_2_score);
  reader_nc_delta.AddVariable("lol_2_score",&tagger.lol_2_score);
  reader_nc_delta.AddVariable("lol_1_score",&tagger.lol_1_score);
  reader_nc_delta.AddVariable("br3_6_score",&tagger.br3_6_score);
  reader_nc_delta.AddVariable("br3_5_score",&tagger.br3_5_score);
  reader_nc_delta.AddVariable("br3_3_score",&tagger.br3_3_score);

  reader_nc_delta.AddVariable("kine_reco_add_energy",&kine.kine_reco_add_energy);
  reader_nc_delta.AddVariable("kine_pio_mass",&kine.kine_pio_mass);

  float temp_kine_pio_flag;
  reader_nc_delta.AddVariable("kine_pio_flag",&temp_kine_pio_flag);

  reader_nc_delta.AddVariable("kine_pio_vtx_dis",&kine.kine_pio_vtx_dis);
  reader_nc_delta.AddVariable("kine_pio_energy_1",&kine.kine_pio_energy_1);
  reader_nc_delta.AddVariable("kine_pio_theta_1",&kine.kine_pio_theta_1);
  reader_nc_delta.AddVariable("kine_pio_phi_1",&kine.kine_pio_phi_1);
  reader_nc_delta.AddVariable("kine_pio_dis_1",&kine.kine_pio_dis_1);
  reader_nc_delta.AddVariable("kine_pio_energy_2",&kine.kine_pio_energy_2);
  reader_nc_delta.AddVariable("kine_pio_theta_2",&kine.kine_pio_theta_2);
  reader_nc_delta.AddVariable("kine_pio_phi_2",&kine.kine_pio_phi_2);
  reader_nc_delta.AddVariable("kine_pio_dis_2",&kine.kine_pio_dis_2);
  reader_nc_delta.AddVariable("kine_pio_angle",&kine.kine_pio_angle);

  //  reader_nc_delta.BookMVA( "MyBDT", "weights/NC_Delta_final_weights.xml");
  reader_nc_delta.BookMVA( "MyBDT", "weights/NC_Delta_prev_training_plus_even_subrun_ncpi0_ncdelta.xml");

  // NC delta 0 track
  TMVA::Reader reader_nc_delta_0track;

  reader_nc_delta_0track.AddVariable("numu_cc_flag_3",&tagger.numu_cc_flag_3);
  reader_nc_delta_0track.AddVariable("numu_cc_3_particle_type",&tagger.numu_cc_3_particle_type);
  reader_nc_delta_0track.AddVariable("numu_cc_3_max_length",&tagger.numu_cc_3_max_length);
  reader_nc_delta_0track.AddVariable("numu_cc_3_track_length",&tagger.numu_cc_3_acc_track_length);
  reader_nc_delta_0track.AddVariable("numu_cc_3_max_length_all",&tagger.numu_cc_3_max_length_all);
  reader_nc_delta_0track.AddVariable("numu_cc_3_max_muon_length",&tagger.numu_cc_3_max_muon_length);
  reader_nc_delta_0track.AddVariable("numu_cc_3_n_daughter_tracks",&tagger.numu_cc_3_n_daughter_tracks);
  reader_nc_delta_0track.AddVariable("numu_cc_3_n_daughter_all",&tagger.numu_cc_3_n_daughter_all);
  reader_nc_delta_0track.AddVariable("cosmict_flag_2",&tagger.cosmict_flag_2);
  reader_nc_delta_0track.AddVariable("cosmict_2_filled",&tagger.cosmict_2_filled);
  reader_nc_delta_0track.AddVariable("cosmict_2_particle_type",&tagger.cosmict_2_particle_type);
  reader_nc_delta_0track.AddVariable("cosmict_2_n_muon_tracks",&tagger.cosmict_2_n_muon_tracks);
  reader_nc_delta_0track.AddVariable("cosmict_2_total_shower_length",&tagger.cosmict_2_total_shower_length);
  reader_nc_delta_0track.AddVariable("cosmict_2_flag_inside",&tagger.cosmict_2_flag_inside);
  reader_nc_delta_0track.AddVariable("cosmict_2_angle_beam",&tagger.cosmict_2_angle_beam);
  reader_nc_delta_0track.AddVariable("cosmict_2_flag_dir_weak",&tagger.cosmict_2_flag_dir_weak);
  reader_nc_delta_0track.AddVariable("cosmict_2_dQ_dx_end",&tagger.cosmict_2_dQ_dx_end);
  reader_nc_delta_0track.AddVariable("cosmict_2_dQ_dx_front",&tagger.cosmict_2_dQ_dx_front);
  reader_nc_delta_0track.AddVariable("cosmict_2_theta",&tagger.cosmict_2_theta);
  reader_nc_delta_0track.AddVariable("cosmict_2_phi",&tagger.cosmict_2_phi);
  reader_nc_delta_0track.AddVariable("cosmict_2_valid_tracks",&tagger.cosmict_2_valid_tracks);
  reader_nc_delta_0track.AddVariable("cosmict_flag_4",&tagger.cosmict_flag_4);
  reader_nc_delta_0track.AddVariable("cosmict_4_filled",&tagger.cosmict_4_filled);
  reader_nc_delta_0track.AddVariable("cosmict_4_flag_inside",&tagger.cosmict_4_flag_inside);
  reader_nc_delta_0track.AddVariable("cosmict_4_angle_beam",&tagger.cosmict_4_angle_beam);
  reader_nc_delta_0track.AddVariable("cosmict_4_connected_showers",&tagger.cosmict_4_connected_showers);
  reader_nc_delta_0track.AddVariable("cosmict_flag_3",&tagger.cosmict_flag_3);
  reader_nc_delta_0track.AddVariable("cosmict_3_filled",&tagger.cosmict_3_filled);
  reader_nc_delta_0track.AddVariable("cosmict_3_flag_inside",&tagger.cosmict_3_flag_inside);
  reader_nc_delta_0track.AddVariable("cosmict_3_angle_beam",&tagger.cosmict_3_angle_beam);
  reader_nc_delta_0track.AddVariable("cosmict_3_flag_dir_weak",&tagger.cosmict_3_flag_dir_weak);
  reader_nc_delta_0track.AddVariable("cosmict_3_dQ_dx_end",&tagger.cosmict_3_dQ_dx_end);
  reader_nc_delta_0track.AddVariable("cosmict_3_dQ_dx_front",&tagger.cosmict_3_dQ_dx_front);
  reader_nc_delta_0track.AddVariable("cosmict_3_theta",&tagger.cosmict_3_theta);
  reader_nc_delta_0track.AddVariable("cosmict_3_phi",&tagger.cosmict_3_phi);
  reader_nc_delta_0track.AddVariable("cosmict_3_valid_tracks",&tagger.cosmict_3_valid_tracks);
  reader_nc_delta_0track.AddVariable("cosmict_flag_5",&tagger.cosmict_flag_5);
  reader_nc_delta_0track.AddVariable("cosmict_5_filled",&tagger.cosmict_5_filled);
  reader_nc_delta_0track.AddVariable("cosmict_5_flag_inside",&tagger.cosmict_5_flag_inside);
  reader_nc_delta_0track.AddVariable("cosmict_5_angle_beam",&tagger.cosmict_5_angle_beam);
  reader_nc_delta_0track.AddVariable("cosmict_5_connected_showers",&tagger.cosmict_5_connected_showers);
  reader_nc_delta_0track.AddVariable("cosmict_flag_6",&tagger.cosmict_flag_6);
  reader_nc_delta_0track.AddVariable("cosmict_6_filled",&tagger.cosmict_6_filled);
  reader_nc_delta_0track.AddVariable("cosmict_6_flag_dir_weak",&tagger.cosmict_6_flag_dir_weak);
  reader_nc_delta_0track.AddVariable("cosmict_6_flag_inside",&tagger.cosmict_6_flag_inside);
  reader_nc_delta_0track.AddVariable("cosmict_6_angle",&tagger.cosmict_6_angle);
  reader_nc_delta_0track.AddVariable("cosmict_flag_7",&tagger.cosmict_flag_7);
  reader_nc_delta_0track.AddVariable("cosmict_7_filled",&tagger.cosmict_7_filled);
  reader_nc_delta_0track.AddVariable("cosmict_7_flag_sec",&tagger.cosmict_7_flag_sec);
  reader_nc_delta_0track.AddVariable("cosmict_7_n_muon_tracks",&tagger.cosmict_7_n_muon_tracks);
  reader_nc_delta_0track.AddVariable("cosmict_7_total_shower_length",&tagger.cosmict_7_total_shower_length);
  reader_nc_delta_0track.AddVariable("cosmict_7_flag_inside",&tagger.cosmict_7_flag_inside);
  reader_nc_delta_0track.AddVariable("cosmict_7_angle_beam",&tagger.cosmict_7_angle_beam);
  reader_nc_delta_0track.AddVariable("cosmict_7_flag_dir_weak",&tagger.cosmict_7_flag_dir_weak);
  reader_nc_delta_0track.AddVariable("cosmict_7_dQ_dx_end",&tagger.cosmict_7_dQ_dx_end);
  reader_nc_delta_0track.AddVariable("cosmict_7_dQ_dx_front",&tagger.cosmict_7_dQ_dx_front);
  reader_nc_delta_0track.AddVariable("cosmict_7_theta",&tagger.cosmict_7_theta);
  reader_nc_delta_0track.AddVariable("cosmict_7_phi",&tagger.cosmict_7_phi);
  reader_nc_delta_0track.AddVariable("cosmict_flag_8",&tagger.cosmict_flag_8);
  reader_nc_delta_0track.AddVariable("cosmict_8_filled",&tagger.cosmict_8_filled);
  reader_nc_delta_0track.AddVariable("cosmict_8_flag_out",&tagger.cosmict_8_flag_out);
  reader_nc_delta_0track.AddVariable("cosmict_8_muon_length",&tagger.cosmict_8_muon_length);
  reader_nc_delta_0track.AddVariable("cosmict_8_acc_length",&tagger.cosmict_8_acc_length);
  reader_nc_delta_0track.AddVariable("cosmict_flag_9",&tagger.cosmict_flag_9);
  reader_nc_delta_0track.AddVariable("cosmic_flag",&tagger.cosmic_flag);
  reader_nc_delta_0track.AddVariable("cosmic_filled",&tagger.cosmic_filled);
  reader_nc_delta_0track.AddVariable("cosmict_flag",&tagger.cosmict_flag);
  reader_nc_delta_0track.AddVariable("numu_cc_flag",&tagger.numu_cc_flag);
  reader_nc_delta_0track.AddVariable("cosmict_flag_1",&tagger.cosmict_flag_1);
  reader_nc_delta_0track.AddVariable("cme_mu_energy",&tagger.cme_mu_energy);
  reader_nc_delta_0track.AddVariable("cme_energy",&tagger.cme_energy);
  reader_nc_delta_0track.AddVariable("cme_mu_length",&tagger.cme_mu_length);
  reader_nc_delta_0track.AddVariable("cme_length",&tagger.cme_length);
  reader_nc_delta_0track.AddVariable("cme_angle_beam",&tagger.cme_angle_beam);
  reader_nc_delta_0track.AddVariable("anc_angle",&tagger.anc_angle);
  reader_nc_delta_0track.AddVariable("anc_max_angle",&tagger.anc_max_angle);
  reader_nc_delta_0track.AddVariable("anc_max_length",&tagger.anc_max_length);
  reader_nc_delta_0track.AddVariable("anc_acc_forward_length",&tagger.anc_acc_forward_length);
  reader_nc_delta_0track.AddVariable("anc_acc_backward_length",&tagger.anc_acc_backward_length);
  reader_nc_delta_0track.AddVariable("anc_acc_forward_length1",&tagger.anc_acc_forward_length1);
  reader_nc_delta_0track.AddVariable("anc_shower_main_length",&tagger.anc_shower_main_length);
  reader_nc_delta_0track.AddVariable("anc_shower_total_length",&tagger.anc_shower_total_length);
  reader_nc_delta_0track.AddVariable("anc_flag_main_outside",&tagger.anc_flag_main_outside);
  reader_nc_delta_0track.AddVariable("gap_flag_prolong_u",&tagger.gap_flag_prolong_u);
  reader_nc_delta_0track.AddVariable("gap_flag_prolong_v",&tagger.gap_flag_prolong_v);
  reader_nc_delta_0track.AddVariable("gap_flag_prolong_w",&tagger.gap_flag_prolong_w);
  reader_nc_delta_0track.AddVariable("gap_flag_parallel",&tagger.gap_flag_parallel);
  reader_nc_delta_0track.AddVariable("gap_n_points",&tagger.gap_n_points);
  reader_nc_delta_0track.AddVariable("gap_n_bad",&tagger.gap_n_bad);
  reader_nc_delta_0track.AddVariable("gap_energy",&tagger.gap_energy);
  reader_nc_delta_0track.AddVariable("gap_num_valid_tracks",&tagger.gap_num_valid_tracks);
  reader_nc_delta_0track.AddVariable("gap_flag_single_shower",&tagger.gap_flag_single_shower);
  reader_nc_delta_0track.AddVariable("hol_1_n_valid_tracks",&tagger.hol_1_n_valid_tracks);
  reader_nc_delta_0track.AddVariable("hol_1_min_angle",&tagger.hol_1_min_angle);
  reader_nc_delta_0track.AddVariable("hol_1_energy",&tagger.hol_1_energy);
  reader_nc_delta_0track.AddVariable("hol_1_flag_all_shower",&tagger.hol_1_flag_all_shower);
  reader_nc_delta_0track.AddVariable("hol_1_min_length",&tagger.hol_1_min_length);
  reader_nc_delta_0track.AddVariable("hol_2_min_angle",&tagger.hol_2_min_angle);
  reader_nc_delta_0track.AddVariable("hol_2_medium_dQ_dx",&tagger.hol_2_medium_dQ_dx);
  reader_nc_delta_0track.AddVariable("hol_2_ncount",&tagger.hol_2_ncount);
  reader_nc_delta_0track.AddVariable("lol_3_angle_beam",&tagger.lol_3_angle_beam);
  reader_nc_delta_0track.AddVariable("lol_3_n_valid_tracks",&tagger.lol_3_n_valid_tracks);
  reader_nc_delta_0track.AddVariable("lol_3_min_angle",&tagger.lol_3_min_angle);
  reader_nc_delta_0track.AddVariable("lol_3_vtx_n_segs",&tagger.lol_3_vtx_n_segs);
  reader_nc_delta_0track.AddVariable("lol_3_shower_main_length",&tagger.lol_3_shower_main_length);
  reader_nc_delta_0track.AddVariable("lol_3_n_out",&tagger.lol_3_n_out);
  reader_nc_delta_0track.AddVariable("lol_3_n_sum",&tagger.lol_3_n_sum);
  reader_nc_delta_0track.AddVariable("mgo_energy",&tagger.mgo_energy);
  reader_nc_delta_0track.AddVariable("mgo_max_energy",&tagger.mgo_max_energy);
  reader_nc_delta_0track.AddVariable("mgo_total_energy",&tagger.mgo_total_energy);
  reader_nc_delta_0track.AddVariable("mgo_n_showers",&tagger.mgo_n_showers);
  reader_nc_delta_0track.AddVariable("mgo_max_energy_1",&tagger.mgo_max_energy_1);
  reader_nc_delta_0track.AddVariable("mgo_max_energy_2",&tagger.mgo_max_energy_2);
  reader_nc_delta_0track.AddVariable("mgo_total_other_energy",&tagger.mgo_total_other_energy);
  reader_nc_delta_0track.AddVariable("mgo_n_total_showers",&tagger.mgo_n_total_showers);
  reader_nc_delta_0track.AddVariable("mgo_total_other_energy_1",&tagger.mgo_total_other_energy_1);
  reader_nc_delta_0track.AddVariable("mgt_flag_single_shower",&tagger.mgt_flag_single_shower);
  reader_nc_delta_0track.AddVariable("mgt_max_energy",&tagger.mgt_max_energy);
  reader_nc_delta_0track.AddVariable("mgt_total_other_energy",&tagger.mgt_total_other_energy);
  reader_nc_delta_0track.AddVariable("mgt_max_energy_1",&tagger.mgt_max_energy_1);
  reader_nc_delta_0track.AddVariable("mgt_e_indirect_max_energy",&tagger.mgt_e_indirect_max_energy);
  reader_nc_delta_0track.AddVariable("mgt_e_direct_max_energy",&tagger.mgt_e_direct_max_energy);
  reader_nc_delta_0track.AddVariable("mgt_n_direct_showers",&tagger.mgt_n_direct_showers);
  reader_nc_delta_0track.AddVariable("mgt_e_direct_total_energy",&tagger.mgt_e_direct_total_energy);
  reader_nc_delta_0track.AddVariable("mgt_flag_indirect_max_pio",&tagger.mgt_flag_indirect_max_pio);
  reader_nc_delta_0track.AddVariable("mgt_e_indirect_total_energy",&tagger.mgt_e_indirect_total_energy);
  reader_nc_delta_0track.AddVariable("mip_quality_energy",&tagger.mip_quality_energy);
  reader_nc_delta_0track.AddVariable("mip_quality_overlap",&tagger.mip_quality_overlap);
  reader_nc_delta_0track.AddVariable("mip_quality_n_showers",&tagger.mip_quality_n_showers);
  reader_nc_delta_0track.AddVariable("mip_quality_n_tracks",&tagger.mip_quality_n_tracks);
  reader_nc_delta_0track.AddVariable("mip_quality_flag_inside_pi0",&tagger.mip_quality_flag_inside_pi0);
  reader_nc_delta_0track.AddVariable("mip_quality_n_pi0_showers",&tagger.mip_quality_n_pi0_showers);
  reader_nc_delta_0track.AddVariable("mip_quality_shortest_length",&tagger.mip_quality_shortest_length);
  reader_nc_delta_0track.AddVariable("mip_quality_acc_length",&tagger.mip_quality_acc_length);
  reader_nc_delta_0track.AddVariable("mip_quality_shortest_angle",&tagger.mip_quality_shortest_angle);
  reader_nc_delta_0track.AddVariable("mip_quality_flag_proton",&tagger.mip_quality_flag_proton);
  reader_nc_delta_0track.AddVariable("br1_1_shower_type",&tagger.br1_1_shower_type);
  reader_nc_delta_0track.AddVariable("br1_1_vtx_n_segs",&tagger.br1_1_vtx_n_segs);
  reader_nc_delta_0track.AddVariable("br1_1_energy",&tagger.br1_1_energy);
  reader_nc_delta_0track.AddVariable("br1_1_n_segs",&tagger.br1_1_n_segs);
  reader_nc_delta_0track.AddVariable("br1_1_flag_sg_topology",&tagger.br1_1_flag_sg_topology);
  reader_nc_delta_0track.AddVariable("br1_1_flag_sg_trajectory",&tagger.br1_1_flag_sg_trajectory);
  reader_nc_delta_0track.AddVariable("br1_1_sg_length",&tagger.br1_1_sg_length);
  reader_nc_delta_0track.AddVariable("br1_2_n_connected",&tagger.br1_2_n_connected);
  reader_nc_delta_0track.AddVariable("br1_2_max_length",&tagger.br1_2_max_length);
  reader_nc_delta_0track.AddVariable("br1_2_n_connected_1",&tagger.br1_2_n_connected_1);
  reader_nc_delta_0track.AddVariable("br1_2_n_shower_segs",&tagger.br1_2_n_shower_segs);
  reader_nc_delta_0track.AddVariable("br1_2_max_length_ratio",&tagger.br1_2_max_length_ratio);
  reader_nc_delta_0track.AddVariable("br1_2_shower_length",&tagger.br1_2_shower_length);
  reader_nc_delta_0track.AddVariable("br1_3_n_connected_p",&tagger.br1_3_n_connected_p);
  reader_nc_delta_0track.AddVariable("br1_3_max_length_p",&tagger.br1_3_max_length_p);
  reader_nc_delta_0track.AddVariable("br1_3_n_shower_main_segs",&tagger.br1_3_n_shower_main_segs);
  reader_nc_delta_0track.AddVariable("br3_1_energy",&tagger.br3_1_energy);
  reader_nc_delta_0track.AddVariable("br3_1_n_shower_segments",&tagger.br3_1_n_shower_segments);
  reader_nc_delta_0track.AddVariable("br3_1_sg_flag_trajectory",&tagger.br3_1_sg_flag_trajectory);
  reader_nc_delta_0track.AddVariable("br3_1_sg_direct_length",&tagger.br3_1_sg_direct_length);
  reader_nc_delta_0track.AddVariable("br3_1_sg_length",&tagger.br3_1_sg_length);
  reader_nc_delta_0track.AddVariable("br3_1_total_main_length",&tagger.br3_1_total_main_length);
  reader_nc_delta_0track.AddVariable("br3_1_total_length",&tagger.br3_1_total_length);
  reader_nc_delta_0track.AddVariable("br3_1_iso_angle",&tagger.br3_1_iso_angle);
  reader_nc_delta_0track.AddVariable("br3_1_sg_flag_topology",&tagger.br3_1_sg_flag_topology);
  reader_nc_delta_0track.AddVariable("br3_2_n_ele",&tagger.br3_2_n_ele);
  reader_nc_delta_0track.AddVariable("br3_2_n_other",&tagger.br3_2_n_other);
  reader_nc_delta_0track.AddVariable("br3_2_other_fid",&tagger.br3_2_other_fid);
  reader_nc_delta_0track.AddVariable("br3_4_acc_length",&tagger.br3_4_acc_length);
  reader_nc_delta_0track.AddVariable("br3_4_total_length",&tagger.br3_4_total_length);
  reader_nc_delta_0track.AddVariable("br3_7_min_angle",&tagger.br3_7_min_angle);
  reader_nc_delta_0track.AddVariable("br3_8_max_dQ_dx",&tagger.br3_8_max_dQ_dx);
  reader_nc_delta_0track.AddVariable("br3_8_n_main_segs",&tagger.br3_8_n_main_segs);
  reader_nc_delta_0track.AddVariable("br4_1_shower_main_length",&tagger.br4_1_shower_main_length);
  reader_nc_delta_0track.AddVariable("br4_1_shower_total_length",&tagger.br4_1_shower_total_length);
  reader_nc_delta_0track.AddVariable("br4_1_min_dis",&tagger.br4_1_min_dis);
  reader_nc_delta_0track.AddVariable("br4_1_energy",&tagger.br4_1_energy);
  reader_nc_delta_0track.AddVariable("br4_1_flag_avoid_muon_check",&tagger.br4_1_flag_avoid_muon_check);
  reader_nc_delta_0track.AddVariable("br4_1_n_vtx_segs",&tagger.br4_1_n_vtx_segs);
  reader_nc_delta_0track.AddVariable("br4_1_n_main_segs",&tagger.br4_1_n_main_segs);
  reader_nc_delta_0track.AddVariable("br4_2_ratio_45",&tagger.br4_2_ratio_45);
  reader_nc_delta_0track.AddVariable("br4_2_ratio_35",&tagger.br4_2_ratio_35);
  reader_nc_delta_0track.AddVariable("br4_2_ratio_25",&tagger.br4_2_ratio_25);
  reader_nc_delta_0track.AddVariable("br4_2_ratio_15",&tagger.br4_2_ratio_15);
  reader_nc_delta_0track.AddVariable("br4_2_ratio1_45",&tagger.br4_2_ratio1_45);
  reader_nc_delta_0track.AddVariable("br4_2_ratio1_35",&tagger.br4_2_ratio1_35);
  reader_nc_delta_0track.AddVariable("br4_2_ratio1_25",&tagger.br4_2_ratio1_25);
  reader_nc_delta_0track.AddVariable("br4_2_ratio1_15",&tagger.br4_2_ratio1_15);
  reader_nc_delta_0track.AddVariable("br4_2_iso_angle",&tagger.br4_2_iso_angle);
  reader_nc_delta_0track.AddVariable("br4_2_iso_angle1",&tagger.br4_2_iso_angle1);
  reader_nc_delta_0track.AddVariable("br4_2_angle",&tagger.br4_2_angle);
  reader_nc_delta_0track.AddVariable("tro_3_stem_length",&tagger.tro_3_stem_length);
  reader_nc_delta_0track.AddVariable("tro_3_n_muon_segs",&tagger.tro_3_n_muon_segs);
  reader_nc_delta_0track.AddVariable("stem_dir_flag_single_shower",&tagger.stem_dir_flag_single_shower);
  reader_nc_delta_0track.AddVariable("stem_dir_angle",&tagger.stem_dir_angle);
  reader_nc_delta_0track.AddVariable("stem_dir_energy",&tagger.stem_dir_energy);
  reader_nc_delta_0track.AddVariable("stem_dir_angle1",&tagger.stem_dir_angle1);
  reader_nc_delta_0track.AddVariable("stem_dir_angle2",&tagger.stem_dir_angle2);
  reader_nc_delta_0track.AddVariable("stem_dir_angle3",&tagger.stem_dir_angle3);
  reader_nc_delta_0track.AddVariable("stem_dir_ratio",&tagger.stem_dir_ratio);
  reader_nc_delta_0track.AddVariable("br2_num_valid_tracks",&tagger.br2_num_valid_tracks);
  reader_nc_delta_0track.AddVariable("br2_n_shower_main_segs",&tagger.br2_n_shower_main_segs);
  reader_nc_delta_0track.AddVariable("br2_max_angle",&tagger.br2_max_angle);
  reader_nc_delta_0track.AddVariable("br2_sg_length",&tagger.br2_sg_length);
  reader_nc_delta_0track.AddVariable("br2_flag_sg_trajectory",&tagger.br2_flag_sg_trajectory);
  reader_nc_delta_0track.AddVariable("stem_len_energy",&tagger.stem_len_energy);
  reader_nc_delta_0track.AddVariable("stem_len_length",&tagger.stem_len_length);
  reader_nc_delta_0track.AddVariable("stem_len_flag_avoid_muon_check",&tagger.stem_len_flag_avoid_muon_check);
  reader_nc_delta_0track.AddVariable("stem_len_num_daughters",&tagger.stem_len_num_daughters);
  reader_nc_delta_0track.AddVariable("stem_len_daughter_length",&tagger.stem_len_daughter_length);
  reader_nc_delta_0track.AddVariable("brm_n_mu_segs",&tagger.brm_n_mu_segs);
  reader_nc_delta_0track.AddVariable("brm_Ep",&tagger.brm_Ep);
  reader_nc_delta_0track.AddVariable("brm_acc_length",&tagger.brm_acc_length);
  reader_nc_delta_0track.AddVariable("brm_shower_total_length",&tagger.brm_shower_total_length);
  reader_nc_delta_0track.AddVariable("brm_connected_length",&tagger.brm_connected_length);
  reader_nc_delta_0track.AddVariable("brm_n_size",&tagger.brm_n_size);
  reader_nc_delta_0track.AddVariable("brm_acc_direct_length",&tagger.brm_acc_direct_length);
  reader_nc_delta_0track.AddVariable("brm_n_shower_main_segs",&tagger.brm_n_shower_main_segs);
  reader_nc_delta_0track.AddVariable("brm_n_mu_main",&tagger.brm_n_mu_main);
  reader_nc_delta_0track.AddVariable("lem_shower_main_length",&tagger.lem_shower_main_length);
  reader_nc_delta_0track.AddVariable("lem_n_3seg",&tagger.lem_n_3seg);
  reader_nc_delta_0track.AddVariable("lem_e_charge",&tagger.lem_e_charge);
  reader_nc_delta_0track.AddVariable("lem_e_dQdx",&tagger.lem_e_dQdx);
  reader_nc_delta_0track.AddVariable("lem_shower_num_main_segs",&tagger.lem_shower_num_main_segs);
  reader_nc_delta_0track.AddVariable("stw_1_energy",&tagger.stw_1_energy);
  reader_nc_delta_0track.AddVariable("stw_1_dis",&tagger.stw_1_dis);
  reader_nc_delta_0track.AddVariable("stw_1_dQ_dx",&tagger.stw_1_dQ_dx);
  reader_nc_delta_0track.AddVariable("stw_1_flag_single_shower",&tagger.stw_1_flag_single_shower);
  reader_nc_delta_0track.AddVariable("stw_1_n_pi0",&tagger.stw_1_n_pi0);
  reader_nc_delta_0track.AddVariable("stw_1_num_valid_tracks",&tagger.stw_1_num_valid_tracks);
  reader_nc_delta_0track.AddVariable("spt_shower_main_length",&tagger.spt_shower_main_length);
  reader_nc_delta_0track.AddVariable("spt_shower_total_length",&tagger.spt_shower_total_length);
  reader_nc_delta_0track.AddVariable("spt_angle_beam",&tagger.spt_angle_beam);
  reader_nc_delta_0track.AddVariable("spt_angle_vertical",&tagger.spt_angle_vertical);
  reader_nc_delta_0track.AddVariable("spt_max_dQ_dx",&tagger.spt_max_dQ_dx);
  reader_nc_delta_0track.AddVariable("spt_angle_beam_1",&tagger.spt_angle_beam_1);
  reader_nc_delta_0track.AddVariable("spt_angle_drift",&tagger.spt_angle_drift);
  reader_nc_delta_0track.AddVariable("spt_angle_drift_1",&tagger.spt_angle_drift_1);
  reader_nc_delta_0track.AddVariable("spt_num_valid_tracks",&tagger.spt_num_valid_tracks);
  reader_nc_delta_0track.AddVariable("spt_n_vtx_segs",&tagger.spt_n_vtx_segs);
  reader_nc_delta_0track.AddVariable("spt_max_length",&tagger.spt_max_length);
  reader_nc_delta_0track.AddVariable("mip_energy",&tagger.mip_energy);
  reader_nc_delta_0track.AddVariable("mip_n_end_reduction",&tagger.mip_n_end_reduction);
  reader_nc_delta_0track.AddVariable("mip_n_first_mip",&tagger.mip_n_first_mip);
  reader_nc_delta_0track.AddVariable("mip_n_first_non_mip",&tagger.mip_n_first_non_mip);
  reader_nc_delta_0track.AddVariable("mip_n_first_non_mip_1",&tagger.mip_n_first_non_mip_1);
  reader_nc_delta_0track.AddVariable("mip_n_first_non_mip_2",&tagger.mip_n_first_non_mip_2);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_0",&tagger.mip_vec_dQ_dx_0);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_1",&tagger.mip_vec_dQ_dx_1);
  reader_nc_delta_0track.AddVariable("mip_max_dQ_dx_sample",&tagger.mip_max_dQ_dx_sample);
  reader_nc_delta_0track.AddVariable("mip_n_below_threshold",&tagger.mip_n_below_threshold);
  reader_nc_delta_0track.AddVariable("mip_n_below_zero",&tagger.mip_n_below_zero);
  reader_nc_delta_0track.AddVariable("mip_n_lowest",&tagger.mip_n_lowest);
  reader_nc_delta_0track.AddVariable("mip_n_highest",&tagger.mip_n_highest);
  reader_nc_delta_0track.AddVariable("mip_lowest_dQ_dx",&tagger.mip_lowest_dQ_dx);
  reader_nc_delta_0track.AddVariable("mip_highest_dQ_dx",&tagger.mip_highest_dQ_dx);
  reader_nc_delta_0track.AddVariable("mip_medium_dQ_dx",&tagger.mip_medium_dQ_dx);
  reader_nc_delta_0track.AddVariable("mip_stem_length",&tagger.mip_stem_length);
  reader_nc_delta_0track.AddVariable("mip_length_main",&tagger.mip_length_main);
  reader_nc_delta_0track.AddVariable("mip_length_total",&tagger.mip_length_total);
  reader_nc_delta_0track.AddVariable("mip_angle_beam",&tagger.mip_angle_beam);
  reader_nc_delta_0track.AddVariable("mip_iso_angle",&tagger.mip_iso_angle);
  reader_nc_delta_0track.AddVariable("mip_n_vertex",&tagger.mip_n_vertex);
  reader_nc_delta_0track.AddVariable("mip_n_good_tracks",&tagger.mip_n_good_tracks);
  reader_nc_delta_0track.AddVariable("mip_E_indirect_max_energy",&tagger.mip_E_indirect_max_energy);
  reader_nc_delta_0track.AddVariable("mip_flag_all_above",&tagger.mip_flag_all_above);
  reader_nc_delta_0track.AddVariable("mip_min_dQ_dx_5",&tagger.mip_min_dQ_dx_5);
  reader_nc_delta_0track.AddVariable("mip_n_other_vertex",&tagger.mip_n_other_vertex);
  reader_nc_delta_0track.AddVariable("mip_n_stem_size",&tagger.mip_n_stem_size);
  reader_nc_delta_0track.AddVariable("mip_flag_stem_trajectory",&tagger.mip_flag_stem_trajectory);
  reader_nc_delta_0track.AddVariable("mip_min_dis",&tagger.mip_min_dis);
  reader_nc_delta_0track.AddVariable("vis_1_n_vtx_segs",&tagger.vis_1_n_vtx_segs);
  reader_nc_delta_0track.AddVariable("vis_1_energy",&tagger.vis_1_energy);
  reader_nc_delta_0track.AddVariable("vis_1_num_good_tracks",&tagger.vis_1_num_good_tracks);
  reader_nc_delta_0track.AddVariable("vis_1_max_angle",&tagger.vis_1_max_angle);
  reader_nc_delta_0track.AddVariable("vis_1_max_shower_angle",&tagger.vis_1_max_shower_angle);
  reader_nc_delta_0track.AddVariable("vis_1_tmp_length1",&tagger.vis_1_tmp_length1);
  reader_nc_delta_0track.AddVariable("vis_1_tmp_length2",&tagger.vis_1_tmp_length2);
  reader_nc_delta_0track.AddVariable("vis_2_n_vtx_segs",&tagger.vis_2_n_vtx_segs);
  reader_nc_delta_0track.AddVariable("vis_2_min_angle",&tagger.vis_2_min_angle);
  reader_nc_delta_0track.AddVariable("vis_2_min_weak_track",&tagger.vis_2_min_weak_track);
  reader_nc_delta_0track.AddVariable("vis_2_angle_beam",&tagger.vis_2_angle_beam);
  reader_nc_delta_0track.AddVariable("vis_2_min_angle1",&tagger.vis_2_min_angle1);
  reader_nc_delta_0track.AddVariable("vis_2_iso_angle1",&tagger.vis_2_iso_angle1);
  reader_nc_delta_0track.AddVariable("vis_2_min_medium_dQ_dx",&tagger.vis_2_min_medium_dQ_dx);
  reader_nc_delta_0track.AddVariable("vis_2_min_length",&tagger.vis_2_min_length);
  reader_nc_delta_0track.AddVariable("vis_2_sg_length",&tagger.vis_2_sg_length);
  reader_nc_delta_0track.AddVariable("vis_2_max_angle",&tagger.vis_2_max_angle);
  reader_nc_delta_0track.AddVariable("vis_2_max_weak_track",&tagger.vis_2_max_weak_track);
  reader_nc_delta_0track.AddVariable("pio_1_mass",&tagger.pio_1_mass);
  reader_nc_delta_0track.AddVariable("pio_1_pio_type",&tagger.pio_1_pio_type);
  reader_nc_delta_0track.AddVariable("pio_1_energy_1",&tagger.pio_1_energy_1);
  reader_nc_delta_0track.AddVariable("pio_1_energy_2",&tagger.pio_1_energy_2);
  reader_nc_delta_0track.AddVariable("pio_1_dis_1",&tagger.pio_1_dis_1);
  reader_nc_delta_0track.AddVariable("pio_1_dis_2",&tagger.pio_1_dis_2);
  reader_nc_delta_0track.AddVariable("pio_mip_id",&tagger.pio_mip_id);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_2",&tagger.mip_vec_dQ_dx_2);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_3",&tagger.mip_vec_dQ_dx_3);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_4",&tagger.mip_vec_dQ_dx_4);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_5",&tagger.mip_vec_dQ_dx_5);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_6",&tagger.mip_vec_dQ_dx_6);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_7",&tagger.mip_vec_dQ_dx_7);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_8",&tagger.mip_vec_dQ_dx_8);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_9",&tagger.mip_vec_dQ_dx_9);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_10",&tagger.mip_vec_dQ_dx_10);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_11",&tagger.mip_vec_dQ_dx_11);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_12",&tagger.mip_vec_dQ_dx_12);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_13",&tagger.mip_vec_dQ_dx_13);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_14",&tagger.mip_vec_dQ_dx_14);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_15",&tagger.mip_vec_dQ_dx_15);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_16",&tagger.mip_vec_dQ_dx_16);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_17",&tagger.mip_vec_dQ_dx_17);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_18",&tagger.mip_vec_dQ_dx_18);
  reader_nc_delta_0track.AddVariable("mip_vec_dQ_dx_19",&tagger.mip_vec_dQ_dx_19);
  reader_nc_delta_0track.AddVariable("cosmict_10_score",&tagger.cosmict_10_score);
  reader_nc_delta_0track.AddVariable("numu_1_score",&tagger.numu_1_score);
  reader_nc_delta_0track.AddVariable("numu_2_score",&tagger.numu_2_score);
  reader_nc_delta_0track.AddVariable("tro_5_score",&tagger.tro_5_score);
  reader_nc_delta_0track.AddVariable("tro_4_score",&tagger.tro_4_score);
  reader_nc_delta_0track.AddVariable("tro_2_score",&tagger.tro_2_score);
  reader_nc_delta_0track.AddVariable("tro_1_score",&tagger.tro_1_score);
  reader_nc_delta_0track.AddVariable("stw_4_score",&tagger.stw_4_score);
  reader_nc_delta_0track.AddVariable("stw_3_score",&tagger.stw_3_score);
  reader_nc_delta_0track.AddVariable("stw_2_score",&tagger.stw_2_score);
  reader_nc_delta_0track.AddVariable("sig_2_score",&tagger.sig_2_score);
  reader_nc_delta_0track.AddVariable("sig_1_score",&tagger.sig_1_score);
  reader_nc_delta_0track.AddVariable("pio_2_score",&tagger.pio_2_score);
  reader_nc_delta_0track.AddVariable("lol_2_score",&tagger.lol_2_score);
  reader_nc_delta_0track.AddVariable("lol_1_score",&tagger.lol_1_score);
  reader_nc_delta_0track.AddVariable("br3_6_score",&tagger.br3_6_score);
  reader_nc_delta_0track.AddVariable("br3_5_score",&tagger.br3_5_score);
  reader_nc_delta_0track.AddVariable("br3_3_score",&tagger.br3_3_score);

  reader_nc_delta_0track.AddVariable("kine_reco_add_energy",&kine.kine_reco_add_energy);
  reader_nc_delta_0track.AddVariable("kine_pio_mass",&kine.kine_pio_mass);

  reader_nc_delta_0track.AddVariable("kine_pio_flag",&temp_kine_pio_flag);

  reader_nc_delta_0track.AddVariable("kine_pio_vtx_dis",&kine.kine_pio_vtx_dis);
  reader_nc_delta_0track.AddVariable("kine_pio_energy_1",&kine.kine_pio_energy_1);
  reader_nc_delta_0track.AddVariable("kine_pio_theta_1",&kine.kine_pio_theta_1);
  reader_nc_delta_0track.AddVariable("kine_pio_phi_1",&kine.kine_pio_phi_1);
  reader_nc_delta_0track.AddVariable("kine_pio_dis_1",&kine.kine_pio_dis_1);
  reader_nc_delta_0track.AddVariable("kine_pio_energy_2",&kine.kine_pio_energy_2);
  reader_nc_delta_0track.AddVariable("kine_pio_theta_2",&kine.kine_pio_theta_2);
  reader_nc_delta_0track.AddVariable("kine_pio_phi_2",&kine.kine_pio_phi_2);
  reader_nc_delta_0track.AddVariable("kine_pio_dis_2",&kine.kine_pio_dis_2);
  reader_nc_delta_0track.AddVariable("kine_pio_angle",&kine.kine_pio_angle);

  //  reader_nc_delta_0track.BookMVA( "MyBDT", "weights/NC_Delta_final_weights.xml");
  reader_nc_delta_0track.BookMVA( "MyBDT", "weights/NC_delta_0_track_final.xml");

  // NC delta N track
  TMVA::Reader reader_nc_delta_ntrack;

  reader_nc_delta_ntrack.AddVariable("numu_cc_flag_3",&tagger.numu_cc_flag_3);
  reader_nc_delta_ntrack.AddVariable("numu_cc_3_particle_type",&tagger.numu_cc_3_particle_type);
  reader_nc_delta_ntrack.AddVariable("numu_cc_3_max_length",&tagger.numu_cc_3_max_length);
  reader_nc_delta_ntrack.AddVariable("numu_cc_3_track_length",&tagger.numu_cc_3_acc_track_length);
  reader_nc_delta_ntrack.AddVariable("numu_cc_3_max_length_all",&tagger.numu_cc_3_max_length_all);
  reader_nc_delta_ntrack.AddVariable("numu_cc_3_max_muon_length",&tagger.numu_cc_3_max_muon_length);
  reader_nc_delta_ntrack.AddVariable("numu_cc_3_n_daughter_tracks",&tagger.numu_cc_3_n_daughter_tracks);
  reader_nc_delta_ntrack.AddVariable("numu_cc_3_n_daughter_all",&tagger.numu_cc_3_n_daughter_all);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_2",&tagger.cosmict_flag_2);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_filled",&tagger.cosmict_2_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_particle_type",&tagger.cosmict_2_particle_type);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_n_muon_tracks",&tagger.cosmict_2_n_muon_tracks);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_total_shower_length",&tagger.cosmict_2_total_shower_length);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_flag_inside",&tagger.cosmict_2_flag_inside);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_angle_beam",&tagger.cosmict_2_angle_beam);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_flag_dir_weak",&tagger.cosmict_2_flag_dir_weak);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_dQ_dx_end",&tagger.cosmict_2_dQ_dx_end);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_dQ_dx_front",&tagger.cosmict_2_dQ_dx_front);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_theta",&tagger.cosmict_2_theta);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_phi",&tagger.cosmict_2_phi);
  reader_nc_delta_ntrack.AddVariable("cosmict_2_valid_tracks",&tagger.cosmict_2_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_4",&tagger.cosmict_flag_4);
  reader_nc_delta_ntrack.AddVariable("cosmict_4_filled",&tagger.cosmict_4_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_4_flag_inside",&tagger.cosmict_4_flag_inside);
  reader_nc_delta_ntrack.AddVariable("cosmict_4_angle_beam",&tagger.cosmict_4_angle_beam);
  reader_nc_delta_ntrack.AddVariable("cosmict_4_connected_showers",&tagger.cosmict_4_connected_showers);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_3",&tagger.cosmict_flag_3);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_filled",&tagger.cosmict_3_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_flag_inside",&tagger.cosmict_3_flag_inside);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_angle_beam",&tagger.cosmict_3_angle_beam);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_flag_dir_weak",&tagger.cosmict_3_flag_dir_weak);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_dQ_dx_end",&tagger.cosmict_3_dQ_dx_end);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_dQ_dx_front",&tagger.cosmict_3_dQ_dx_front);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_theta",&tagger.cosmict_3_theta);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_phi",&tagger.cosmict_3_phi);
  reader_nc_delta_ntrack.AddVariable("cosmict_3_valid_tracks",&tagger.cosmict_3_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_5",&tagger.cosmict_flag_5);
  reader_nc_delta_ntrack.AddVariable("cosmict_5_filled",&tagger.cosmict_5_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_5_flag_inside",&tagger.cosmict_5_flag_inside);
  reader_nc_delta_ntrack.AddVariable("cosmict_5_angle_beam",&tagger.cosmict_5_angle_beam);
  reader_nc_delta_ntrack.AddVariable("cosmict_5_connected_showers",&tagger.cosmict_5_connected_showers);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_6",&tagger.cosmict_flag_6);
  reader_nc_delta_ntrack.AddVariable("cosmict_6_filled",&tagger.cosmict_6_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_6_flag_dir_weak",&tagger.cosmict_6_flag_dir_weak);
  reader_nc_delta_ntrack.AddVariable("cosmict_6_flag_inside",&tagger.cosmict_6_flag_inside);
  reader_nc_delta_ntrack.AddVariable("cosmict_6_angle",&tagger.cosmict_6_angle);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_7",&tagger.cosmict_flag_7);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_filled",&tagger.cosmict_7_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_flag_sec",&tagger.cosmict_7_flag_sec);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_n_muon_tracks",&tagger.cosmict_7_n_muon_tracks);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_total_shower_length",&tagger.cosmict_7_total_shower_length);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_flag_inside",&tagger.cosmict_7_flag_inside);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_angle_beam",&tagger.cosmict_7_angle_beam);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_flag_dir_weak",&tagger.cosmict_7_flag_dir_weak);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_dQ_dx_end",&tagger.cosmict_7_dQ_dx_end);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_dQ_dx_front",&tagger.cosmict_7_dQ_dx_front);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_theta",&tagger.cosmict_7_theta);
  reader_nc_delta_ntrack.AddVariable("cosmict_7_phi",&tagger.cosmict_7_phi);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_8",&tagger.cosmict_flag_8);
  reader_nc_delta_ntrack.AddVariable("cosmict_8_filled",&tagger.cosmict_8_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_8_flag_out",&tagger.cosmict_8_flag_out);
  reader_nc_delta_ntrack.AddVariable("cosmict_8_muon_length",&tagger.cosmict_8_muon_length);
  reader_nc_delta_ntrack.AddVariable("cosmict_8_acc_length",&tagger.cosmict_8_acc_length);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_9",&tagger.cosmict_flag_9);
  reader_nc_delta_ntrack.AddVariable("cosmic_flag",&tagger.cosmic_flag);
  reader_nc_delta_ntrack.AddVariable("cosmic_filled",&tagger.cosmic_filled);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag",&tagger.cosmict_flag);
  reader_nc_delta_ntrack.AddVariable("numu_cc_flag",&tagger.numu_cc_flag);
  reader_nc_delta_ntrack.AddVariable("cosmict_flag_1",&tagger.cosmict_flag_1);
  reader_nc_delta_ntrack.AddVariable("cme_mu_energy",&tagger.cme_mu_energy);
  reader_nc_delta_ntrack.AddVariable("cme_energy",&tagger.cme_energy);
  reader_nc_delta_ntrack.AddVariable("cme_mu_length",&tagger.cme_mu_length);
  reader_nc_delta_ntrack.AddVariable("cme_length",&tagger.cme_length);
  reader_nc_delta_ntrack.AddVariable("cme_angle_beam",&tagger.cme_angle_beam);
  reader_nc_delta_ntrack.AddVariable("anc_angle",&tagger.anc_angle);
  reader_nc_delta_ntrack.AddVariable("anc_max_angle",&tagger.anc_max_angle);
  reader_nc_delta_ntrack.AddVariable("anc_max_length",&tagger.anc_max_length);
  reader_nc_delta_ntrack.AddVariable("anc_acc_forward_length",&tagger.anc_acc_forward_length);
  reader_nc_delta_ntrack.AddVariable("anc_acc_backward_length",&tagger.anc_acc_backward_length);
  reader_nc_delta_ntrack.AddVariable("anc_acc_forward_length1",&tagger.anc_acc_forward_length1);
  reader_nc_delta_ntrack.AddVariable("anc_shower_main_length",&tagger.anc_shower_main_length);
  reader_nc_delta_ntrack.AddVariable("anc_shower_total_length",&tagger.anc_shower_total_length);
  reader_nc_delta_ntrack.AddVariable("anc_flag_main_outside",&tagger.anc_flag_main_outside);
  reader_nc_delta_ntrack.AddVariable("gap_flag_prolong_u",&tagger.gap_flag_prolong_u);
  reader_nc_delta_ntrack.AddVariable("gap_flag_prolong_v",&tagger.gap_flag_prolong_v);
  reader_nc_delta_ntrack.AddVariable("gap_flag_prolong_w",&tagger.gap_flag_prolong_w);
  reader_nc_delta_ntrack.AddVariable("gap_flag_parallel",&tagger.gap_flag_parallel);
  reader_nc_delta_ntrack.AddVariable("gap_n_points",&tagger.gap_n_points);
  reader_nc_delta_ntrack.AddVariable("gap_n_bad",&tagger.gap_n_bad);
  reader_nc_delta_ntrack.AddVariable("gap_energy",&tagger.gap_energy);
  reader_nc_delta_ntrack.AddVariable("gap_num_valid_tracks",&tagger.gap_num_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("gap_flag_single_shower",&tagger.gap_flag_single_shower);
  reader_nc_delta_ntrack.AddVariable("hol_1_n_valid_tracks",&tagger.hol_1_n_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("hol_1_min_angle",&tagger.hol_1_min_angle);
  reader_nc_delta_ntrack.AddVariable("hol_1_energy",&tagger.hol_1_energy);
  reader_nc_delta_ntrack.AddVariable("hol_1_flag_all_shower",&tagger.hol_1_flag_all_shower);
  reader_nc_delta_ntrack.AddVariable("hol_1_min_length",&tagger.hol_1_min_length);
  reader_nc_delta_ntrack.AddVariable("hol_2_min_angle",&tagger.hol_2_min_angle);
  reader_nc_delta_ntrack.AddVariable("hol_2_medium_dQ_dx",&tagger.hol_2_medium_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("hol_2_ncount",&tagger.hol_2_ncount);
  reader_nc_delta_ntrack.AddVariable("lol_3_angle_beam",&tagger.lol_3_angle_beam);
  reader_nc_delta_ntrack.AddVariable("lol_3_n_valid_tracks",&tagger.lol_3_n_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("lol_3_min_angle",&tagger.lol_3_min_angle);
  reader_nc_delta_ntrack.AddVariable("lol_3_vtx_n_segs",&tagger.lol_3_vtx_n_segs);
  reader_nc_delta_ntrack.AddVariable("lol_3_shower_main_length",&tagger.lol_3_shower_main_length);
  reader_nc_delta_ntrack.AddVariable("lol_3_n_out",&tagger.lol_3_n_out);
  reader_nc_delta_ntrack.AddVariable("lol_3_n_sum",&tagger.lol_3_n_sum);
  reader_nc_delta_ntrack.AddVariable("mgo_energy",&tagger.mgo_energy);
  reader_nc_delta_ntrack.AddVariable("mgo_max_energy",&tagger.mgo_max_energy);
  reader_nc_delta_ntrack.AddVariable("mgo_total_energy",&tagger.mgo_total_energy);
  reader_nc_delta_ntrack.AddVariable("mgo_n_showers",&tagger.mgo_n_showers);
  reader_nc_delta_ntrack.AddVariable("mgo_max_energy_1",&tagger.mgo_max_energy_1);
  reader_nc_delta_ntrack.AddVariable("mgo_max_energy_2",&tagger.mgo_max_energy_2);
  reader_nc_delta_ntrack.AddVariable("mgo_total_other_energy",&tagger.mgo_total_other_energy);
  reader_nc_delta_ntrack.AddVariable("mgo_n_total_showers",&tagger.mgo_n_total_showers);
  reader_nc_delta_ntrack.AddVariable("mgo_total_other_energy_1",&tagger.mgo_total_other_energy_1);
  reader_nc_delta_ntrack.AddVariable("mgt_flag_single_shower",&tagger.mgt_flag_single_shower);
  reader_nc_delta_ntrack.AddVariable("mgt_max_energy",&tagger.mgt_max_energy);
  reader_nc_delta_ntrack.AddVariable("mgt_total_other_energy",&tagger.mgt_total_other_energy);
  reader_nc_delta_ntrack.AddVariable("mgt_max_energy_1",&tagger.mgt_max_energy_1);
  reader_nc_delta_ntrack.AddVariable("mgt_e_indirect_max_energy",&tagger.mgt_e_indirect_max_energy);
  reader_nc_delta_ntrack.AddVariable("mgt_e_direct_max_energy",&tagger.mgt_e_direct_max_energy);
  reader_nc_delta_ntrack.AddVariable("mgt_n_direct_showers",&tagger.mgt_n_direct_showers);
  reader_nc_delta_ntrack.AddVariable("mgt_e_direct_total_energy",&tagger.mgt_e_direct_total_energy);
  reader_nc_delta_ntrack.AddVariable("mgt_flag_indirect_max_pio",&tagger.mgt_flag_indirect_max_pio);
  reader_nc_delta_ntrack.AddVariable("mgt_e_indirect_total_energy",&tagger.mgt_e_indirect_total_energy);
  reader_nc_delta_ntrack.AddVariable("mip_quality_energy",&tagger.mip_quality_energy);
  reader_nc_delta_ntrack.AddVariable("mip_quality_overlap",&tagger.mip_quality_overlap);
  reader_nc_delta_ntrack.AddVariable("mip_quality_n_showers",&tagger.mip_quality_n_showers);
  reader_nc_delta_ntrack.AddVariable("mip_quality_n_tracks",&tagger.mip_quality_n_tracks);
  reader_nc_delta_ntrack.AddVariable("mip_quality_flag_inside_pi0",&tagger.mip_quality_flag_inside_pi0);
  reader_nc_delta_ntrack.AddVariable("mip_quality_n_pi0_showers",&tagger.mip_quality_n_pi0_showers);
  reader_nc_delta_ntrack.AddVariable("mip_quality_shortest_length",&tagger.mip_quality_shortest_length);
  reader_nc_delta_ntrack.AddVariable("mip_quality_acc_length",&tagger.mip_quality_acc_length);
  reader_nc_delta_ntrack.AddVariable("mip_quality_shortest_angle",&tagger.mip_quality_shortest_angle);
  reader_nc_delta_ntrack.AddVariable("mip_quality_flag_proton",&tagger.mip_quality_flag_proton);
  reader_nc_delta_ntrack.AddVariable("br1_1_shower_type",&tagger.br1_1_shower_type);
  reader_nc_delta_ntrack.AddVariable("br1_1_vtx_n_segs",&tagger.br1_1_vtx_n_segs);
  reader_nc_delta_ntrack.AddVariable("br1_1_energy",&tagger.br1_1_energy);
  reader_nc_delta_ntrack.AddVariable("br1_1_n_segs",&tagger.br1_1_n_segs);
  reader_nc_delta_ntrack.AddVariable("br1_1_flag_sg_topology",&tagger.br1_1_flag_sg_topology);
  reader_nc_delta_ntrack.AddVariable("br1_1_flag_sg_trajectory",&tagger.br1_1_flag_sg_trajectory);
  reader_nc_delta_ntrack.AddVariable("br1_1_sg_length",&tagger.br1_1_sg_length);
  reader_nc_delta_ntrack.AddVariable("br1_2_n_connected",&tagger.br1_2_n_connected);
  reader_nc_delta_ntrack.AddVariable("br1_2_max_length",&tagger.br1_2_max_length);
  reader_nc_delta_ntrack.AddVariable("br1_2_n_connected_1",&tagger.br1_2_n_connected_1);
  reader_nc_delta_ntrack.AddVariable("br1_2_n_shower_segs",&tagger.br1_2_n_shower_segs);
  reader_nc_delta_ntrack.AddVariable("br1_2_max_length_ratio",&tagger.br1_2_max_length_ratio);
  reader_nc_delta_ntrack.AddVariable("br1_2_shower_length",&tagger.br1_2_shower_length);
  reader_nc_delta_ntrack.AddVariable("br1_3_n_connected_p",&tagger.br1_3_n_connected_p);
  reader_nc_delta_ntrack.AddVariable("br1_3_max_length_p",&tagger.br1_3_max_length_p);
  reader_nc_delta_ntrack.AddVariable("br1_3_n_shower_main_segs",&tagger.br1_3_n_shower_main_segs);
  reader_nc_delta_ntrack.AddVariable("br3_1_energy",&tagger.br3_1_energy);
  reader_nc_delta_ntrack.AddVariable("br3_1_n_shower_segments",&tagger.br3_1_n_shower_segments);
  reader_nc_delta_ntrack.AddVariable("br3_1_sg_flag_trajectory",&tagger.br3_1_sg_flag_trajectory);
  reader_nc_delta_ntrack.AddVariable("br3_1_sg_direct_length",&tagger.br3_1_sg_direct_length);
  reader_nc_delta_ntrack.AddVariable("br3_1_sg_length",&tagger.br3_1_sg_length);
  reader_nc_delta_ntrack.AddVariable("br3_1_total_main_length",&tagger.br3_1_total_main_length);
  reader_nc_delta_ntrack.AddVariable("br3_1_total_length",&tagger.br3_1_total_length);
  reader_nc_delta_ntrack.AddVariable("br3_1_iso_angle",&tagger.br3_1_iso_angle);
  reader_nc_delta_ntrack.AddVariable("br3_1_sg_flag_topology",&tagger.br3_1_sg_flag_topology);
  reader_nc_delta_ntrack.AddVariable("br3_2_n_ele",&tagger.br3_2_n_ele);
  reader_nc_delta_ntrack.AddVariable("br3_2_n_other",&tagger.br3_2_n_other);
  reader_nc_delta_ntrack.AddVariable("br3_2_other_fid",&tagger.br3_2_other_fid);
  reader_nc_delta_ntrack.AddVariable("br3_4_acc_length",&tagger.br3_4_acc_length);
  reader_nc_delta_ntrack.AddVariable("br3_4_total_length",&tagger.br3_4_total_length);
  reader_nc_delta_ntrack.AddVariable("br3_7_min_angle",&tagger.br3_7_min_angle);
  reader_nc_delta_ntrack.AddVariable("br3_8_max_dQ_dx",&tagger.br3_8_max_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("br3_8_n_main_segs",&tagger.br3_8_n_main_segs);
  reader_nc_delta_ntrack.AddVariable("br4_1_shower_main_length",&tagger.br4_1_shower_main_length);
  reader_nc_delta_ntrack.AddVariable("br4_1_shower_total_length",&tagger.br4_1_shower_total_length);
  reader_nc_delta_ntrack.AddVariable("br4_1_min_dis",&tagger.br4_1_min_dis);
  reader_nc_delta_ntrack.AddVariable("br4_1_energy",&tagger.br4_1_energy);
  reader_nc_delta_ntrack.AddVariable("br4_1_flag_avoid_muon_check",&tagger.br4_1_flag_avoid_muon_check);
  reader_nc_delta_ntrack.AddVariable("br4_1_n_vtx_segs",&tagger.br4_1_n_vtx_segs);
  reader_nc_delta_ntrack.AddVariable("br4_1_n_main_segs",&tagger.br4_1_n_main_segs);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio_45",&tagger.br4_2_ratio_45);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio_35",&tagger.br4_2_ratio_35);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio_25",&tagger.br4_2_ratio_25);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio_15",&tagger.br4_2_ratio_15);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio1_45",&tagger.br4_2_ratio1_45);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio1_35",&tagger.br4_2_ratio1_35);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio1_25",&tagger.br4_2_ratio1_25);
  reader_nc_delta_ntrack.AddVariable("br4_2_ratio1_15",&tagger.br4_2_ratio1_15);
  reader_nc_delta_ntrack.AddVariable("br4_2_iso_angle",&tagger.br4_2_iso_angle);
  reader_nc_delta_ntrack.AddVariable("br4_2_iso_angle1",&tagger.br4_2_iso_angle1);
  reader_nc_delta_ntrack.AddVariable("br4_2_angle",&tagger.br4_2_angle);
  reader_nc_delta_ntrack.AddVariable("tro_3_stem_length",&tagger.tro_3_stem_length);
  reader_nc_delta_ntrack.AddVariable("tro_3_n_muon_segs",&tagger.tro_3_n_muon_segs);
  reader_nc_delta_ntrack.AddVariable("stem_dir_flag_single_shower",&tagger.stem_dir_flag_single_shower);
  reader_nc_delta_ntrack.AddVariable("stem_dir_angle",&tagger.stem_dir_angle);
  reader_nc_delta_ntrack.AddVariable("stem_dir_energy",&tagger.stem_dir_energy);
  reader_nc_delta_ntrack.AddVariable("stem_dir_angle1",&tagger.stem_dir_angle1);
  reader_nc_delta_ntrack.AddVariable("stem_dir_angle2",&tagger.stem_dir_angle2);
  reader_nc_delta_ntrack.AddVariable("stem_dir_angle3",&tagger.stem_dir_angle3);
  reader_nc_delta_ntrack.AddVariable("stem_dir_ratio",&tagger.stem_dir_ratio);
  reader_nc_delta_ntrack.AddVariable("br2_num_valid_tracks",&tagger.br2_num_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("br2_n_shower_main_segs",&tagger.br2_n_shower_main_segs);
  reader_nc_delta_ntrack.AddVariable("br2_max_angle",&tagger.br2_max_angle);
  reader_nc_delta_ntrack.AddVariable("br2_sg_length",&tagger.br2_sg_length);
  reader_nc_delta_ntrack.AddVariable("br2_flag_sg_trajectory",&tagger.br2_flag_sg_trajectory);
  reader_nc_delta_ntrack.AddVariable("stem_len_energy",&tagger.stem_len_energy);
  reader_nc_delta_ntrack.AddVariable("stem_len_length",&tagger.stem_len_length);
  reader_nc_delta_ntrack.AddVariable("stem_len_flag_avoid_muon_check",&tagger.stem_len_flag_avoid_muon_check);
  reader_nc_delta_ntrack.AddVariable("stem_len_num_daughters",&tagger.stem_len_num_daughters);
  reader_nc_delta_ntrack.AddVariable("stem_len_daughter_length",&tagger.stem_len_daughter_length);
  reader_nc_delta_ntrack.AddVariable("brm_n_mu_segs",&tagger.brm_n_mu_segs);
  reader_nc_delta_ntrack.AddVariable("brm_Ep",&tagger.brm_Ep);
  reader_nc_delta_ntrack.AddVariable("brm_acc_length",&tagger.brm_acc_length);
  reader_nc_delta_ntrack.AddVariable("brm_shower_total_length",&tagger.brm_shower_total_length);
  reader_nc_delta_ntrack.AddVariable("brm_connected_length",&tagger.brm_connected_length);
  reader_nc_delta_ntrack.AddVariable("brm_n_size",&tagger.brm_n_size);
  reader_nc_delta_ntrack.AddVariable("brm_acc_direct_length",&tagger.brm_acc_direct_length);
  reader_nc_delta_ntrack.AddVariable("brm_n_shower_main_segs",&tagger.brm_n_shower_main_segs);
  reader_nc_delta_ntrack.AddVariable("brm_n_mu_main",&tagger.brm_n_mu_main);
  reader_nc_delta_ntrack.AddVariable("lem_shower_main_length",&tagger.lem_shower_main_length);
  reader_nc_delta_ntrack.AddVariable("lem_n_3seg",&tagger.lem_n_3seg);
  reader_nc_delta_ntrack.AddVariable("lem_e_charge",&tagger.lem_e_charge);
  reader_nc_delta_ntrack.AddVariable("lem_e_dQdx",&tagger.lem_e_dQdx);
  reader_nc_delta_ntrack.AddVariable("lem_shower_num_main_segs",&tagger.lem_shower_num_main_segs);
  reader_nc_delta_ntrack.AddVariable("stw_1_energy",&tagger.stw_1_energy);
  reader_nc_delta_ntrack.AddVariable("stw_1_dis",&tagger.stw_1_dis);
  reader_nc_delta_ntrack.AddVariable("stw_1_dQ_dx",&tagger.stw_1_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("stw_1_flag_single_shower",&tagger.stw_1_flag_single_shower);
  reader_nc_delta_ntrack.AddVariable("stw_1_n_pi0",&tagger.stw_1_n_pi0);
  reader_nc_delta_ntrack.AddVariable("stw_1_num_valid_tracks",&tagger.stw_1_num_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("spt_shower_main_length",&tagger.spt_shower_main_length);
  reader_nc_delta_ntrack.AddVariable("spt_shower_total_length",&tagger.spt_shower_total_length);
  reader_nc_delta_ntrack.AddVariable("spt_angle_beam",&tagger.spt_angle_beam);
  reader_nc_delta_ntrack.AddVariable("spt_angle_vertical",&tagger.spt_angle_vertical);
  reader_nc_delta_ntrack.AddVariable("spt_max_dQ_dx",&tagger.spt_max_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("spt_angle_beam_1",&tagger.spt_angle_beam_1);
  reader_nc_delta_ntrack.AddVariable("spt_angle_drift",&tagger.spt_angle_drift);
  reader_nc_delta_ntrack.AddVariable("spt_angle_drift_1",&tagger.spt_angle_drift_1);
  reader_nc_delta_ntrack.AddVariable("spt_num_valid_tracks",&tagger.spt_num_valid_tracks);
  reader_nc_delta_ntrack.AddVariable("spt_n_vtx_segs",&tagger.spt_n_vtx_segs);
  reader_nc_delta_ntrack.AddVariable("spt_max_length",&tagger.spt_max_length);
  reader_nc_delta_ntrack.AddVariable("mip_energy",&tagger.mip_energy);
  reader_nc_delta_ntrack.AddVariable("mip_n_end_reduction",&tagger.mip_n_end_reduction);
  reader_nc_delta_ntrack.AddVariable("mip_n_first_mip",&tagger.mip_n_first_mip);
  reader_nc_delta_ntrack.AddVariable("mip_n_first_non_mip",&tagger.mip_n_first_non_mip);
  reader_nc_delta_ntrack.AddVariable("mip_n_first_non_mip_1",&tagger.mip_n_first_non_mip_1);
  reader_nc_delta_ntrack.AddVariable("mip_n_first_non_mip_2",&tagger.mip_n_first_non_mip_2);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_0",&tagger.mip_vec_dQ_dx_0);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_1",&tagger.mip_vec_dQ_dx_1);
  reader_nc_delta_ntrack.AddVariable("mip_max_dQ_dx_sample",&tagger.mip_max_dQ_dx_sample);
  reader_nc_delta_ntrack.AddVariable("mip_n_below_threshold",&tagger.mip_n_below_threshold);
  reader_nc_delta_ntrack.AddVariable("mip_n_below_zero",&tagger.mip_n_below_zero);
  reader_nc_delta_ntrack.AddVariable("mip_n_lowest",&tagger.mip_n_lowest);
  reader_nc_delta_ntrack.AddVariable("mip_n_highest",&tagger.mip_n_highest);
  reader_nc_delta_ntrack.AddVariable("mip_lowest_dQ_dx",&tagger.mip_lowest_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("mip_highest_dQ_dx",&tagger.mip_highest_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("mip_medium_dQ_dx",&tagger.mip_medium_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("mip_stem_length",&tagger.mip_stem_length);
  reader_nc_delta_ntrack.AddVariable("mip_length_main",&tagger.mip_length_main);
  reader_nc_delta_ntrack.AddVariable("mip_length_total",&tagger.mip_length_total);
  reader_nc_delta_ntrack.AddVariable("mip_angle_beam",&tagger.mip_angle_beam);
  reader_nc_delta_ntrack.AddVariable("mip_iso_angle",&tagger.mip_iso_angle);
  reader_nc_delta_ntrack.AddVariable("mip_n_vertex",&tagger.mip_n_vertex);
  reader_nc_delta_ntrack.AddVariable("mip_n_good_tracks",&tagger.mip_n_good_tracks);
  reader_nc_delta_ntrack.AddVariable("mip_E_indirect_max_energy",&tagger.mip_E_indirect_max_energy);
  reader_nc_delta_ntrack.AddVariable("mip_flag_all_above",&tagger.mip_flag_all_above);
  reader_nc_delta_ntrack.AddVariable("mip_min_dQ_dx_5",&tagger.mip_min_dQ_dx_5);
  reader_nc_delta_ntrack.AddVariable("mip_n_other_vertex",&tagger.mip_n_other_vertex);
  reader_nc_delta_ntrack.AddVariable("mip_n_stem_size",&tagger.mip_n_stem_size);
  reader_nc_delta_ntrack.AddVariable("mip_flag_stem_trajectory",&tagger.mip_flag_stem_trajectory);
  reader_nc_delta_ntrack.AddVariable("mip_min_dis",&tagger.mip_min_dis);
  reader_nc_delta_ntrack.AddVariable("vis_1_n_vtx_segs",&tagger.vis_1_n_vtx_segs);
  reader_nc_delta_ntrack.AddVariable("vis_1_energy",&tagger.vis_1_energy);
  reader_nc_delta_ntrack.AddVariable("vis_1_num_good_tracks",&tagger.vis_1_num_good_tracks);
  reader_nc_delta_ntrack.AddVariable("vis_1_max_angle",&tagger.vis_1_max_angle);
  reader_nc_delta_ntrack.AddVariable("vis_1_max_shower_angle",&tagger.vis_1_max_shower_angle);
  reader_nc_delta_ntrack.AddVariable("vis_1_tmp_length1",&tagger.vis_1_tmp_length1);
  reader_nc_delta_ntrack.AddVariable("vis_1_tmp_length2",&tagger.vis_1_tmp_length2);
  reader_nc_delta_ntrack.AddVariable("vis_2_n_vtx_segs",&tagger.vis_2_n_vtx_segs);
  reader_nc_delta_ntrack.AddVariable("vis_2_min_angle",&tagger.vis_2_min_angle);
  reader_nc_delta_ntrack.AddVariable("vis_2_min_weak_track",&tagger.vis_2_min_weak_track);
  reader_nc_delta_ntrack.AddVariable("vis_2_angle_beam",&tagger.vis_2_angle_beam);
  reader_nc_delta_ntrack.AddVariable("vis_2_min_angle1",&tagger.vis_2_min_angle1);
  reader_nc_delta_ntrack.AddVariable("vis_2_iso_angle1",&tagger.vis_2_iso_angle1);
  reader_nc_delta_ntrack.AddVariable("vis_2_min_medium_dQ_dx",&tagger.vis_2_min_medium_dQ_dx);
  reader_nc_delta_ntrack.AddVariable("vis_2_min_length",&tagger.vis_2_min_length);
  reader_nc_delta_ntrack.AddVariable("vis_2_sg_length",&tagger.vis_2_sg_length);
  reader_nc_delta_ntrack.AddVariable("vis_2_max_angle",&tagger.vis_2_max_angle);
  reader_nc_delta_ntrack.AddVariable("vis_2_max_weak_track",&tagger.vis_2_max_weak_track);
  reader_nc_delta_ntrack.AddVariable("pio_1_mass",&tagger.pio_1_mass);
  reader_nc_delta_ntrack.AddVariable("pio_1_pio_type",&tagger.pio_1_pio_type);
  reader_nc_delta_ntrack.AddVariable("pio_1_energy_1",&tagger.pio_1_energy_1);
  reader_nc_delta_ntrack.AddVariable("pio_1_energy_2",&tagger.pio_1_energy_2);
  reader_nc_delta_ntrack.AddVariable("pio_1_dis_1",&tagger.pio_1_dis_1);
  reader_nc_delta_ntrack.AddVariable("pio_1_dis_2",&tagger.pio_1_dis_2);
  reader_nc_delta_ntrack.AddVariable("pio_mip_id",&tagger.pio_mip_id);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_2",&tagger.mip_vec_dQ_dx_2);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_3",&tagger.mip_vec_dQ_dx_3);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_4",&tagger.mip_vec_dQ_dx_4);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_5",&tagger.mip_vec_dQ_dx_5);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_6",&tagger.mip_vec_dQ_dx_6);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_7",&tagger.mip_vec_dQ_dx_7);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_8",&tagger.mip_vec_dQ_dx_8);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_9",&tagger.mip_vec_dQ_dx_9);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_10",&tagger.mip_vec_dQ_dx_10);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_11",&tagger.mip_vec_dQ_dx_11);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_12",&tagger.mip_vec_dQ_dx_12);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_13",&tagger.mip_vec_dQ_dx_13);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_14",&tagger.mip_vec_dQ_dx_14);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_15",&tagger.mip_vec_dQ_dx_15);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_16",&tagger.mip_vec_dQ_dx_16);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_17",&tagger.mip_vec_dQ_dx_17);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_18",&tagger.mip_vec_dQ_dx_18);
  reader_nc_delta_ntrack.AddVariable("mip_vec_dQ_dx_19",&tagger.mip_vec_dQ_dx_19);
  reader_nc_delta_ntrack.AddVariable("cosmict_10_score",&tagger.cosmict_10_score);
  reader_nc_delta_ntrack.AddVariable("numu_1_score",&tagger.numu_1_score);
  reader_nc_delta_ntrack.AddVariable("numu_2_score",&tagger.numu_2_score);
  reader_nc_delta_ntrack.AddVariable("tro_5_score",&tagger.tro_5_score);
  reader_nc_delta_ntrack.AddVariable("tro_4_score",&tagger.tro_4_score);
  reader_nc_delta_ntrack.AddVariable("tro_2_score",&tagger.tro_2_score);
  reader_nc_delta_ntrack.AddVariable("tro_1_score",&tagger.tro_1_score);
  reader_nc_delta_ntrack.AddVariable("stw_4_score",&tagger.stw_4_score);
  reader_nc_delta_ntrack.AddVariable("stw_3_score",&tagger.stw_3_score);
  reader_nc_delta_ntrack.AddVariable("stw_2_score",&tagger.stw_2_score);
  reader_nc_delta_ntrack.AddVariable("sig_2_score",&tagger.sig_2_score);
  reader_nc_delta_ntrack.AddVariable("sig_1_score",&tagger.sig_1_score);
  reader_nc_delta_ntrack.AddVariable("pio_2_score",&tagger.pio_2_score);
  reader_nc_delta_ntrack.AddVariable("lol_2_score",&tagger.lol_2_score);
  reader_nc_delta_ntrack.AddVariable("lol_1_score",&tagger.lol_1_score);
  reader_nc_delta_ntrack.AddVariable("br3_6_score",&tagger.br3_6_score);
  reader_nc_delta_ntrack.AddVariable("br3_5_score",&tagger.br3_5_score);
  reader_nc_delta_ntrack.AddVariable("br3_3_score",&tagger.br3_3_score);

  reader_nc_delta_ntrack.AddVariable("kine_reco_add_energy",&kine.kine_reco_add_energy);
  reader_nc_delta_ntrack.AddVariable("kine_pio_mass",&kine.kine_pio_mass);

  reader_nc_delta_ntrack.AddVariable("kine_pio_flag",&temp_kine_pio_flag);

  reader_nc_delta_ntrack.AddVariable("kine_pio_vtx_dis",&kine.kine_pio_vtx_dis);
  reader_nc_delta_ntrack.AddVariable("kine_pio_energy_1",&kine.kine_pio_energy_1);
  reader_nc_delta_ntrack.AddVariable("kine_pio_theta_1",&kine.kine_pio_theta_1);
  reader_nc_delta_ntrack.AddVariable("kine_pio_phi_1",&kine.kine_pio_phi_1);
  reader_nc_delta_ntrack.AddVariable("kine_pio_dis_1",&kine.kine_pio_dis_1);
  reader_nc_delta_ntrack.AddVariable("kine_pio_energy_2",&kine.kine_pio_energy_2);
  reader_nc_delta_ntrack.AddVariable("kine_pio_theta_2",&kine.kine_pio_theta_2);
  reader_nc_delta_ntrack.AddVariable("kine_pio_phi_2",&kine.kine_pio_phi_2);
  reader_nc_delta_ntrack.AddVariable("kine_pio_dis_2",&kine.kine_pio_dis_2);
  reader_nc_delta_ntrack.AddVariable("kine_pio_angle",&kine.kine_pio_angle);

  //  reader_nc_delta_ntrack.BookMVA( "MyBDT", "weights/NC_Delta_final_weights.xml");
  reader_nc_delta_ntrack.BookMVA( "MyBDT", "weights/NC_delta_N_track_final.xml");


  // Now NCpio by Giacomo @ Yale
  TMVA::Reader reader_nc_pi0;
  reader_nc_pi0.AddVariable("cosmic_n_solid_tracks",&tagger.cosmic_n_solid_tracks);
  reader_nc_pi0.AddVariable("cosmic_energy_main_showers",&tagger.cosmic_energy_main_showers);
  reader_nc_pi0.AddVariable("cosmic_energy_direct_showers",&tagger.cosmic_energy_direct_showers);
  reader_nc_pi0.AddVariable("cosmic_energy_indirect_showers",&tagger.cosmic_energy_indirect_showers);
  reader_nc_pi0.AddVariable("cosmic_n_direct_showers",&tagger.cosmic_n_direct_showers);
  reader_nc_pi0.AddVariable("cosmic_n_indirect_showers",&tagger.cosmic_n_indirect_showers);
  reader_nc_pi0.AddVariable("cosmic_n_main_showers",&tagger.cosmic_n_main_showers);
  reader_nc_pi0.AddVariable("gap_flag_prolong_u",&tagger.gap_flag_prolong_u);
  reader_nc_pi0.AddVariable("gap_flag_prolong_v",&tagger.gap_flag_prolong_v);
  reader_nc_pi0.AddVariable("gap_flag_prolong_w",&tagger.gap_flag_prolong_w);
  reader_nc_pi0.AddVariable("gap_flag_parallel",&tagger.gap_flag_parallel);
  reader_nc_pi0.AddVariable("gap_n_points",&tagger.gap_n_points);
  reader_nc_pi0.AddVariable("gap_n_bad",&tagger.gap_n_bad);
  reader_nc_pi0.AddVariable("gap_energy",&tagger.gap_energy);
  reader_nc_pi0.AddVariable("gap_num_valid_tracks",&tagger.gap_num_valid_tracks);
  reader_nc_pi0.AddVariable("gap_flag_single_shower",&tagger.gap_flag_single_shower);
  reader_nc_pi0.AddVariable("mip_quality_overlap",&tagger.mip_quality_overlap);
  reader_nc_pi0.AddVariable("mip_quality_n_showers",&tagger.mip_quality_n_showers);
  reader_nc_pi0.AddVariable("mip_quality_n_tracks",&tagger.mip_quality_n_tracks);
  reader_nc_pi0.AddVariable("mip_quality_flag_inside_pi0",&tagger.mip_quality_flag_inside_pi0);
  reader_nc_pi0.AddVariable("mip_quality_n_pi0_showers",&tagger.mip_quality_n_pi0_showers);
  reader_nc_pi0.AddVariable("mip_quality_shortest_length",&tagger.mip_quality_shortest_length);
  reader_nc_pi0.AddVariable("mip_quality_acc_length",&tagger.mip_quality_acc_length);
  reader_nc_pi0.AddVariable("mip_quality_shortest_angle",&tagger.mip_quality_shortest_angle);
  reader_nc_pi0.AddVariable("mip_quality_flag_proton",&tagger.mip_quality_flag_proton);
  reader_nc_pi0.AddVariable("mip_energy",&tagger.mip_energy);
  reader_nc_pi0.AddVariable("mip_n_end_reduction",&tagger.mip_n_end_reduction);
  reader_nc_pi0.AddVariable("mip_n_first_mip",&tagger.mip_n_first_mip);
  reader_nc_pi0.AddVariable("mip_n_first_non_mip",&tagger.mip_n_first_non_mip);
  reader_nc_pi0.AddVariable("mip_n_first_non_mip_1",&tagger.mip_n_first_non_mip_1);
  reader_nc_pi0.AddVariable("mip_n_first_non_mip_2",&tagger.mip_n_first_non_mip_2);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_0",&tagger.mip_vec_dQ_dx_0);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_1",&tagger.mip_vec_dQ_dx_1);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_2",&tagger.mip_vec_dQ_dx_2);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_3",&tagger.mip_vec_dQ_dx_3);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_4",&tagger.mip_vec_dQ_dx_4);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_5",&tagger.mip_vec_dQ_dx_5);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_6",&tagger.mip_vec_dQ_dx_6);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_7",&tagger.mip_vec_dQ_dx_7);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_8",&tagger.mip_vec_dQ_dx_8);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_9",&tagger.mip_vec_dQ_dx_9);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_10",&tagger.mip_vec_dQ_dx_10);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_11",&tagger.mip_vec_dQ_dx_11);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_12",&tagger.mip_vec_dQ_dx_12);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_13",&tagger.mip_vec_dQ_dx_13);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_14",&tagger.mip_vec_dQ_dx_14);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_15",&tagger.mip_vec_dQ_dx_15);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_16",&tagger.mip_vec_dQ_dx_16);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_17",&tagger.mip_vec_dQ_dx_17);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_18",&tagger.mip_vec_dQ_dx_18);
  reader_nc_pi0.AddVariable("mip_vec_dQ_dx_19",&tagger.mip_vec_dQ_dx_19);
  reader_nc_pi0.AddVariable("mip_max_dQ_dx_sample",&tagger.mip_max_dQ_dx_sample);
  reader_nc_pi0.AddVariable("mip_n_below_threshold",&tagger.mip_n_below_threshold);
  reader_nc_pi0.AddVariable("mip_n_below_zero",&tagger.mip_n_below_zero);
  reader_nc_pi0.AddVariable("mip_n_lowest",&tagger.mip_n_lowest);
  reader_nc_pi0.AddVariable("mip_n_highest",&tagger.mip_n_highest);
  reader_nc_pi0.AddVariable("mip_lowest_dQ_dx",&tagger.mip_lowest_dQ_dx);
  reader_nc_pi0.AddVariable("mip_highest_dQ_dx",&tagger.mip_highest_dQ_dx);
  reader_nc_pi0.AddVariable("mip_medium_dQ_dx",&tagger.mip_medium_dQ_dx);
  reader_nc_pi0.AddVariable("mip_stem_length",&tagger.mip_stem_length);
  reader_nc_pi0.AddVariable("mip_length_main",&tagger.mip_length_main);
  reader_nc_pi0.AddVariable("mip_length_total",&tagger.mip_length_total);
  reader_nc_pi0.AddVariable("mip_angle_beam",&tagger.mip_angle_beam);
  reader_nc_pi0.AddVariable("mip_iso_angle",&tagger.mip_iso_angle);
  reader_nc_pi0.AddVariable("mip_n_vertex",&tagger.mip_n_vertex);
  reader_nc_pi0.AddVariable("mip_n_good_tracks",&tagger.mip_n_good_tracks);
  reader_nc_pi0.AddVariable("mip_E_indirect_max_energy",&tagger.mip_E_indirect_max_energy);
  reader_nc_pi0.AddVariable("mip_flag_all_above",&tagger.mip_flag_all_above);
  reader_nc_pi0.AddVariable("mip_min_dQ_dx_5",&tagger.mip_min_dQ_dx_5);
  reader_nc_pi0.AddVariable("mip_n_other_vertex",&tagger.mip_n_other_vertex);
  reader_nc_pi0.AddVariable("mip_n_stem_size",&tagger.mip_n_stem_size);
  reader_nc_pi0.AddVariable("mip_flag_stem_trajectory",&tagger.mip_flag_stem_trajectory);
  reader_nc_pi0.AddVariable("mip_min_dis",&tagger.mip_min_dis);
  reader_nc_pi0.AddVariable("pio_mip_id",&tagger.pio_mip_id);
  reader_nc_pi0.AddVariable("pio_flag_pio",&tagger.pio_flag_pio);
  reader_nc_pi0.AddVariable("pio_1_mass",&tagger.pio_1_mass);
  reader_nc_pi0.AddVariable("pio_1_pio_type",&tagger.pio_1_pio_type);
  reader_nc_pi0.AddVariable("pio_1_energy_1",&tagger.pio_1_energy_1);
  reader_nc_pi0.AddVariable("pio_1_energy_2",&tagger.pio_1_energy_2);
  reader_nc_pi0.AddVariable("pio_1_dis_1",&tagger.pio_1_dis_1);
  reader_nc_pi0.AddVariable("pio_1_dis_2",&tagger.pio_1_dis_2);
  reader_nc_pi0.AddVariable("mgo_max_energy",&tagger.mgo_max_energy);
  reader_nc_pi0.AddVariable("mgo_total_energy",&tagger.mgo_total_energy);
  reader_nc_pi0.AddVariable("mgo_n_showers",&tagger.mgo_n_showers);
  reader_nc_pi0.AddVariable("mgo_max_energy_1",&tagger.mgo_max_energy_1);
  reader_nc_pi0.AddVariable("mgo_max_energy_2",&tagger.mgo_max_energy_2);
  reader_nc_pi0.AddVariable("mgo_total_other_energy",&tagger.mgo_total_other_energy);
  reader_nc_pi0.AddVariable("mgo_n_total_showers",&tagger.mgo_n_total_showers);
  reader_nc_pi0.AddVariable("mgo_total_other_energy_1",&tagger.mgo_total_other_energy_1);
  reader_nc_pi0.AddVariable("mgt_total_other_energy",&tagger.mgt_total_other_energy);
  reader_nc_pi0.AddVariable("mgt_max_energy_1",&tagger.mgt_max_energy_1);
  reader_nc_pi0.AddVariable("mgt_e_indirect_max_energy",&tagger.mgt_e_indirect_max_energy);
  reader_nc_pi0.AddVariable("mgt_e_direct_max_energy",&tagger.mgt_e_direct_max_energy);
  reader_nc_pi0.AddVariable("mgt_n_direct_showers",&tagger.mgt_n_direct_showers);
  reader_nc_pi0.AddVariable("mgt_e_direct_total_energy",&tagger.mgt_e_direct_total_energy);
  reader_nc_pi0.AddVariable("mgt_flag_indirect_max_pio",&tagger.mgt_flag_indirect_max_pio);
  reader_nc_pi0.AddVariable("mgt_e_indirect_total_energy",&tagger.mgt_e_indirect_total_energy);
  reader_nc_pi0.AddVariable("stw_1_dis",&tagger.stw_1_dis);
  reader_nc_pi0.AddVariable("stw_1_dQ_dx",&tagger.stw_1_dQ_dx);
  reader_nc_pi0.AddVariable("stw_1_n_pi0",&tagger.stw_1_n_pi0);
  reader_nc_pi0.AddVariable("stw_1_num_valid_tracks",&tagger.stw_1_num_valid_tracks);
  reader_nc_pi0.AddVariable("spt_shower_main_length",&tagger.spt_shower_main_length);
  reader_nc_pi0.AddVariable("spt_shower_total_length",&tagger.spt_shower_total_length);
  reader_nc_pi0.AddVariable("spt_angle_beam",&tagger.spt_angle_beam);
  reader_nc_pi0.AddVariable("spt_angle_vertical",&tagger.spt_angle_vertical);
  reader_nc_pi0.AddVariable("spt_angle_beam_1",&tagger.spt_angle_beam_1);
  reader_nc_pi0.AddVariable("spt_angle_drift_1",&tagger.spt_angle_drift_1);
  reader_nc_pi0.AddVariable("spt_num_valid_tracks",&tagger.spt_num_valid_tracks);
  reader_nc_pi0.AddVariable("spt_n_vtx_segs",&tagger.spt_n_vtx_segs);
  reader_nc_pi0.AddVariable("spt_max_length",&tagger.spt_max_length);
  reader_nc_pi0.AddVariable("stem_len_length",&tagger.stem_len_length);
  reader_nc_pi0.AddVariable("stem_len_flag_avoid_muon_check",&tagger.stem_len_flag_avoid_muon_check);
  reader_nc_pi0.AddVariable("stem_len_num_daughters",&tagger.stem_len_num_daughters);
  reader_nc_pi0.AddVariable("stem_len_daughter_length",&tagger.stem_len_daughter_length);
  reader_nc_pi0.AddVariable("lem_n_3seg",&tagger.lem_n_3seg);
  reader_nc_pi0.AddVariable("lem_e_charge",&tagger.lem_e_charge);
  reader_nc_pi0.AddVariable("lem_e_dQdx",&tagger.lem_e_dQdx);
  reader_nc_pi0.AddVariable("lem_shower_num_segs",&tagger.lem_shower_num_segs);
  reader_nc_pi0.AddVariable("lem_shower_num_main_segs",&tagger.lem_shower_num_main_segs);
  reader_nc_pi0.AddVariable("brm_n_mu_segs",&tagger.brm_n_mu_segs);
  reader_nc_pi0.AddVariable("brm_Ep",&tagger.brm_Ep);
  reader_nc_pi0.AddVariable("brm_acc_length",&tagger.brm_acc_length);
  reader_nc_pi0.AddVariable("brm_connected_length",&tagger.brm_connected_length);
  reader_nc_pi0.AddVariable("brm_n_size",&tagger.brm_n_size);
  reader_nc_pi0.AddVariable("brm_acc_direct_length",&tagger.brm_acc_direct_length);
  reader_nc_pi0.AddVariable("brm_n_mu_main",&tagger.brm_n_mu_main);
  reader_nc_pi0.AddVariable("cme_mu_energy",&tagger.cme_mu_energy);
  reader_nc_pi0.AddVariable("cme_mu_length",&tagger.cme_mu_length);
  reader_nc_pi0.AddVariable("cme_angle_beam",&tagger.cme_angle_beam);
  reader_nc_pi0.AddVariable("anc_max_angle",&tagger.anc_max_angle);
  reader_nc_pi0.AddVariable("anc_max_length",&tagger.anc_max_length);
  reader_nc_pi0.AddVariable("anc_acc_forward_length",&tagger.anc_acc_forward_length);
  reader_nc_pi0.AddVariable("anc_acc_backward_length",&tagger.anc_acc_backward_length);
  reader_nc_pi0.AddVariable("anc_acc_forward_length1",&tagger.anc_acc_forward_length1);
  reader_nc_pi0.AddVariable("anc_flag_main_outside",&tagger.anc_flag_main_outside);
  reader_nc_pi0.AddVariable("stem_dir_angle",&tagger.stem_dir_angle);
  reader_nc_pi0.AddVariable("stem_dir_angle1",&tagger.stem_dir_angle1);
  reader_nc_pi0.AddVariable("stem_dir_angle2",&tagger.stem_dir_angle2);
  reader_nc_pi0.AddVariable("stem_dir_angle3",&tagger.stem_dir_angle3);
  reader_nc_pi0.AddVariable("stem_dir_ratio",&tagger.stem_dir_ratio);
  reader_nc_pi0.AddVariable("vis_1_n_vtx_segs",&tagger.vis_1_n_vtx_segs);
  reader_nc_pi0.AddVariable("vis_1_energy",&tagger.vis_1_energy);
  reader_nc_pi0.AddVariable("vis_1_num_good_tracks",&tagger.vis_1_num_good_tracks);
  reader_nc_pi0.AddVariable("vis_1_max_angle",&tagger.vis_1_max_angle);
  reader_nc_pi0.AddVariable("vis_1_max_shower_angle",&tagger.vis_1_max_shower_angle);
  reader_nc_pi0.AddVariable("vis_1_tmp_length1",&tagger.vis_1_tmp_length1);
  reader_nc_pi0.AddVariable("vis_1_particle_type",&tagger.vis_1_particle_type);
  reader_nc_pi0.AddVariable("vis_2_n_vtx_segs",&tagger.vis_2_n_vtx_segs);
  reader_nc_pi0.AddVariable("vis_2_min_angle",&tagger.vis_2_min_angle);
  reader_nc_pi0.AddVariable("vis_2_min_weak_track",&tagger.vis_2_min_weak_track);
  reader_nc_pi0.AddVariable("vis_2_angle_beam",&tagger.vis_2_angle_beam);
  reader_nc_pi0.AddVariable("vis_2_min_angle1",&tagger.vis_2_min_angle1);
  reader_nc_pi0.AddVariable("vis_2_iso_angle1",&tagger.vis_2_iso_angle1);
  reader_nc_pi0.AddVariable("vis_2_min_medium_dQ_dx",&tagger.vis_2_min_medium_dQ_dx);
  reader_nc_pi0.AddVariable("vis_2_min_length",&tagger.vis_2_min_length);
  reader_nc_pi0.AddVariable("vis_2_sg_length",&tagger.vis_2_sg_length);
  reader_nc_pi0.AddVariable("vis_2_max_angle",&tagger.vis_2_max_angle);
  reader_nc_pi0.AddVariable("vis_2_max_weak_track",&tagger.vis_2_max_weak_track);
  reader_nc_pi0.AddVariable("br1_1_shower_type",&tagger.br1_1_shower_type);
  reader_nc_pi0.AddVariable("br1_1_flag_sg_topology",&tagger.br1_1_flag_sg_topology);
  reader_nc_pi0.AddVariable("br1_2_n_connected",&tagger.br1_2_n_connected);
  reader_nc_pi0.AddVariable("br1_2_max_length",&tagger.br1_2_max_length);
  reader_nc_pi0.AddVariable("br1_2_n_connected_1",&tagger.br1_2_n_connected_1);
  reader_nc_pi0.AddVariable("br1_2_max_length_ratio",&tagger.br1_2_max_length_ratio);
  reader_nc_pi0.AddVariable("br1_3_n_connected_p",&tagger.br1_3_n_connected_p);
  reader_nc_pi0.AddVariable("br1_3_max_length_p",&tagger.br1_3_max_length_p);
  reader_nc_pi0.AddVariable("br2_max_angle",&tagger.br2_max_angle);
  reader_nc_pi0.AddVariable("br3_1_sg_direct_length",&tagger.br3_1_sg_direct_length);
  reader_nc_pi0.AddVariable("br3_1_total_length",&tagger.br3_1_total_length);
  reader_nc_pi0.AddVariable("br3_1_iso_angle",&tagger.br3_1_iso_angle);
  reader_nc_pi0.AddVariable("br3_2_n_ele",&tagger.br3_2_n_ele);
  reader_nc_pi0.AddVariable("br3_2_n_other",&tagger.br3_2_n_other);
  reader_nc_pi0.AddVariable("br3_2_other_fid",&tagger.br3_2_other_fid);
  reader_nc_pi0.AddVariable("br3_4_acc_length",&tagger.br3_4_acc_length);
  reader_nc_pi0.AddVariable("br3_7_min_angle",&tagger.br3_7_min_angle);
  reader_nc_pi0.AddVariable("br3_8_max_dQ_dx",&tagger.br3_8_max_dQ_dx);
  reader_nc_pi0.AddVariable("br4_1_min_dis",&tagger.br4_1_min_dis);
  reader_nc_pi0.AddVariable("br4_2_ratio_45",&tagger.br4_2_ratio_45);
  reader_nc_pi0.AddVariable("br4_2_ratio_35",&tagger.br4_2_ratio_35);
  reader_nc_pi0.AddVariable("br4_2_ratio_25",&tagger.br4_2_ratio_25);
  reader_nc_pi0.AddVariable("br4_2_ratio_15",&tagger.br4_2_ratio_15);
  reader_nc_pi0.AddVariable("br4_2_ratio1_45",&tagger.br4_2_ratio1_45);
  reader_nc_pi0.AddVariable("br4_2_ratio1_35",&tagger.br4_2_ratio1_35);
  reader_nc_pi0.AddVariable("br4_2_ratio1_25",&tagger.br4_2_ratio1_25);
  reader_nc_pi0.AddVariable("br4_2_ratio1_15",&tagger.br4_2_ratio1_15);
  reader_nc_pi0.AddVariable("br4_2_iso_angle",&tagger.br4_2_iso_angle);
  reader_nc_pi0.AddVariable("br4_2_iso_angle1",&tagger.br4_2_iso_angle1);
  reader_nc_pi0.AddVariable("br4_2_angle",&tagger.br4_2_angle);
  reader_nc_pi0.AddVariable("tro_3_stem_length",&tagger.tro_3_stem_length);
  reader_nc_pi0.AddVariable("tro_3_n_muon_segs",&tagger.tro_3_n_muon_segs);
  reader_nc_pi0.AddVariable("hol_1_n_valid_tracks",&tagger.hol_1_n_valid_tracks);
  reader_nc_pi0.AddVariable("hol_1_min_angle",&tagger.hol_1_min_angle);
  reader_nc_pi0.AddVariable("hol_1_energy",&tagger.hol_1_energy);
  reader_nc_pi0.AddVariable("hol_1_flag_all_shower",&tagger.hol_1_flag_all_shower);
  reader_nc_pi0.AddVariable("hol_1_min_length",&tagger.hol_1_min_length);
  reader_nc_pi0.AddVariable("hol_2_min_angle",&tagger.hol_2_min_angle);
  reader_nc_pi0.AddVariable("hol_2_medium_dQ_dx",&tagger.hol_2_medium_dQ_dx);
  reader_nc_pi0.AddVariable("hol_2_ncount",&tagger.hol_2_ncount);
  reader_nc_pi0.AddVariable("lol_3_angle_beam",&tagger.lol_3_angle_beam);
  reader_nc_pi0.AddVariable("lol_3_n_valid_tracks",&tagger.lol_3_n_valid_tracks);
  reader_nc_pi0.AddVariable("lol_3_min_angle",&tagger.lol_3_min_angle);
  reader_nc_pi0.AddVariable("lol_3_n_out",&tagger.lol_3_n_out);
  reader_nc_pi0.AddVariable("lol_3_n_sum",&tagger.lol_3_n_sum);
  reader_nc_pi0.AddVariable("cosmict_2_particle_type",&tagger.cosmict_2_particle_type);
  reader_nc_pi0.AddVariable("cosmict_2_n_muon_tracks",&tagger.cosmict_2_n_muon_tracks);
  reader_nc_pi0.AddVariable("cosmict_2_flag_inside",&tagger.cosmict_2_flag_inside);
  reader_nc_pi0.AddVariable("cosmict_2_angle_beam",&tagger.cosmict_2_angle_beam);
  reader_nc_pi0.AddVariable("cosmict_2_flag_dir_weak",&tagger.cosmict_2_flag_dir_weak);
  reader_nc_pi0.AddVariable("cosmict_2_dQ_dx_end",&tagger.cosmict_2_dQ_dx_end);
  reader_nc_pi0.AddVariable("cosmict_2_dQ_dx_front",&tagger.cosmict_2_dQ_dx_front);
  reader_nc_pi0.AddVariable("cosmict_2_phi",&tagger.cosmict_2_phi);
  reader_nc_pi0.AddVariable("cosmict_2_valid_tracks",&tagger.cosmict_2_valid_tracks);
  reader_nc_pi0.AddVariable("cosmict_3_flag_inside",&tagger.cosmict_3_flag_inside);
  reader_nc_pi0.AddVariable("cosmict_3_angle_beam",&tagger.cosmict_3_angle_beam);
  reader_nc_pi0.AddVariable("cosmict_3_flag_dir_weak",&tagger.cosmict_3_flag_dir_weak);
  reader_nc_pi0.AddVariable("cosmict_3_dQ_dx_end",&tagger.cosmict_3_dQ_dx_end);
  reader_nc_pi0.AddVariable("cosmict_3_dQ_dx_front",&tagger.cosmict_3_dQ_dx_front);
  reader_nc_pi0.AddVariable("cosmict_3_phi",&tagger.cosmict_3_phi);
  reader_nc_pi0.AddVariable("cosmict_3_valid_tracks",&tagger.cosmict_3_valid_tracks);
  reader_nc_pi0.AddVariable("cosmict_4_flag_inside",&tagger.cosmict_4_flag_inside);
  reader_nc_pi0.AddVariable("cosmict_6_flag_dir_weak",&tagger.cosmict_6_flag_dir_weak);
  reader_nc_pi0.AddVariable("cosmict_6_flag_inside",&tagger.cosmict_6_flag_inside);
  reader_nc_pi0.AddVariable("cosmict_6_angle",&tagger.cosmict_6_angle);
  reader_nc_pi0.AddVariable("cosmict_7_flag_sec",&tagger.cosmict_7_flag_sec);
  reader_nc_pi0.AddVariable("cosmict_7_n_muon_tracks",&tagger.cosmict_7_n_muon_tracks);
  reader_nc_pi0.AddVariable("cosmict_7_flag_inside",&tagger.cosmict_7_flag_inside);
  reader_nc_pi0.AddVariable("cosmict_7_angle_beam",&tagger.cosmict_7_angle_beam);
  reader_nc_pi0.AddVariable("cosmict_7_flag_dir_weak",&tagger.cosmict_7_flag_dir_weak);
  reader_nc_pi0.AddVariable("cosmict_7_dQ_dx_end",&tagger.cosmict_7_dQ_dx_end);
  reader_nc_pi0.AddVariable("cosmict_7_dQ_dx_front",&tagger.cosmict_7_dQ_dx_front);
  reader_nc_pi0.AddVariable("cosmict_7_phi",&tagger.cosmict_7_phi);
  reader_nc_pi0.AddVariable("cosmict_8_flag_out",&tagger.cosmict_8_flag_out);
  reader_nc_pi0.AddVariable("cosmict_8_muon_length",&tagger.cosmict_8_muon_length);
  reader_nc_pi0.AddVariable("cosmict_8_acc_length",&tagger.cosmict_8_acc_length);
  reader_nc_pi0.AddVariable("numu_cc_3_particle_type",&tagger.numu_cc_3_particle_type);
  reader_nc_pi0.AddVariable("numu_cc_3_max_length",&tagger.numu_cc_3_max_length);
  reader_nc_pi0.AddVariable("numu_cc_3_track_length",&tagger.numu_cc_3_acc_track_length);
  reader_nc_pi0.AddVariable("numu_cc_3_max_length_all",&tagger.numu_cc_3_max_length_all);
  reader_nc_pi0.AddVariable("numu_cc_3_max_muon_length",&tagger.numu_cc_3_max_muon_length);
  reader_nc_pi0.AddVariable("numu_cc_3_n_daughter_all",&tagger.numu_cc_3_n_daughter_all);
  reader_nc_pi0.AddVariable("pio_2_score",&tagger.pio_2_score);
  reader_nc_pi0.AddVariable("sig_1_score",&tagger.sig_1_score);
  reader_nc_pi0.AddVariable("sig_2_score",&tagger.sig_2_score);
  reader_nc_pi0.AddVariable("stw_2_score",&tagger.stw_2_score);
  reader_nc_pi0.AddVariable("stw_3_score",&tagger.stw_3_score);
  reader_nc_pi0.AddVariable("stw_4_score",&tagger.stw_4_score);
  reader_nc_pi0.AddVariable("br3_3_score",&tagger.br3_3_score);
  reader_nc_pi0.AddVariable("br3_5_score",&tagger.br3_5_score);
  reader_nc_pi0.AddVariable("br3_6_score",&tagger.br3_6_score);
  reader_nc_pi0.AddVariable("lol_1_score",&tagger.lol_1_score);
  reader_nc_pi0.AddVariable("lol_2_score",&tagger.lol_2_score);
  reader_nc_pi0.AddVariable("tro_1_score",&tagger.tro_1_score);
  reader_nc_pi0.AddVariable("tro_2_score",&tagger.tro_2_score);
  reader_nc_pi0.AddVariable("tro_4_score",&tagger.tro_4_score);
  reader_nc_pi0.AddVariable("tro_5_score",&tagger.tro_5_score);
  reader_nc_pi0.AddVariable("cosmict_10_score",&tagger.cosmict_10_score);
  reader_nc_pi0.AddVariable("numu_1_score",&tagger.numu_1_score);
  reader_nc_pi0.AddVariable("numu_2_score",&tagger.numu_2_score);
  reader_nc_pi0.AddVariable("numu_score",&tagger.numu_score);
  reader_nc_pi0.AddVariable("nue_score",&tagger.nue_score);

  reader_nc_pi0.AddVariable("kine_reco_Enu",&kine.kine_reco_Enu);
  reader_nc_pi0.AddVariable("kine_reco_add_energy",&kine.kine_reco_add_energy);
  reader_nc_pi0.AddVariable("kine_pio_mass",&kine.kine_pio_mass);

  reader_nc_pi0.AddVariable("kine_pio_flag",&temp_kine_pio_flag);
  //  reader_nc_pi0.AddVariable("kine_pio_flag",&kine.kine_pio_flag);
  reader_nc_pi0.AddVariable("kine_pio_vtx_dis",&kine.kine_pio_vtx_dis);
  reader_nc_pi0.AddVariable("kine_pio_energy_1",&kine.kine_pio_energy_1);
  reader_nc_pi0.AddVariable("kine_pio_theta_1",&kine.kine_pio_theta_1);
  reader_nc_pi0.AddVariable("kine_pio_phi_1",&kine.kine_pio_phi_1);
  reader_nc_pi0.AddVariable("kine_pio_dis_1",&kine.kine_pio_dis_1);
  reader_nc_pi0.AddVariable("kine_pio_energy_2",&kine.kine_pio_energy_2);
  reader_nc_pi0.AddVariable("kine_pio_theta_2",&kine.kine_pio_theta_2);
  reader_nc_pi0.AddVariable("kine_pio_phi_2",&kine.kine_pio_phi_2);
  reader_nc_pi0.AddVariable("kine_pio_dis_2",&kine.kine_pio_dis_2);
  reader_nc_pi0.AddVariable("kine_pio_angle",&kine.kine_pio_angle);
  reader_nc_pi0.BookMVA( "MyBDT", "weights/ncpio_weights_TMVA.xml");



  // Single Photon BDTs ... by Erin @ UCSB
  TMVA::Reader reader_single_photon_numu;
  TMVA::Reader reader_single_photon_other;
  TMVA::Reader reader_single_photon_ncpi0;
  TMVA::Reader reader_single_photon_nue;

  reader_single_photon_numu.AddVariable("numu_cc_flag_3",&tagger.numu_cc_flag_3);
  reader_single_photon_numu.AddVariable("numu_cc_3_particle_type", &tagger.numu_cc_3_particle_type);
  reader_single_photon_numu.AddVariable("numu_cc_3_max_length", &tagger.numu_cc_3_max_length);
  reader_single_photon_numu.AddVariable("numu_cc_3_track_length",&tagger.numu_cc_3_acc_track_length);
  reader_single_photon_numu.AddVariable("numu_cc_3_max_length_all",&tagger.numu_cc_3_max_length_all);
  reader_single_photon_numu.AddVariable("numu_cc_3_max_muon_length",&tagger.numu_cc_3_max_muon_length);
  reader_single_photon_numu.AddVariable("numu_cc_3_n_daughter_tracks",&tagger.numu_cc_3_n_daughter_tracks);
  reader_single_photon_numu.AddVariable("numu_cc_3_n_daughter_all",&tagger.numu_cc_3_n_daughter_all);
  reader_single_photon_numu.AddVariable("shw_sp_br1_1_shower_type",&tagger.shw_sp_br1_1_shower_type);
  reader_single_photon_numu.AddVariable("shw_sp_br1_1_vtx_n_segs",&tagger.shw_sp_br1_1_vtx_n_segs);
  reader_single_photon_numu.AddVariable("shw_sp_br1_1_n_segs",&tagger.shw_sp_br1_1_n_segs);
  reader_single_photon_numu.AddVariable("shw_sp_br1_1_sg_length",&tagger.shw_sp_br1_1_sg_length);
  reader_single_photon_numu.AddVariable("shw_sp_br1_2_n_connected",&tagger.shw_sp_br1_2_n_connected);
  reader_single_photon_numu.AddVariable("shw_sp_br1_2_max_length",&tagger.shw_sp_br1_2_max_length);
  reader_single_photon_numu.AddVariable("shw_sp_br1_2_n_connected_1",&tagger.shw_sp_br1_2_n_connected_1);
  reader_single_photon_numu.AddVariable("shw_sp_br1_2_n_shower_segs",&tagger.shw_sp_br1_2_n_shower_segs);
  reader_single_photon_numu.AddVariable("shw_sp_br1_2_max_length_ratio",&tagger.shw_sp_br1_2_max_length_ratio);
  reader_single_photon_numu.AddVariable("shw_sp_br1_2_shower_length",&tagger.shw_sp_br1_2_shower_length);
  reader_single_photon_numu.AddVariable("shw_sp_br1_3_max_length_p",&tagger.shw_sp_br1_3_max_length_p);
  reader_single_photon_numu.AddVariable("shw_sp_br1_3_n_shower_main_segs",&tagger.shw_sp_br1_3_n_shower_main_segs);
  reader_single_photon_numu.AddVariable("shw_sp_br3_1_n_shower_segments",&tagger.shw_sp_br3_1_n_shower_segments);
  reader_single_photon_numu.AddVariable("shw_sp_br3_1_sg_direct_length",&tagger.shw_sp_br3_1_sg_direct_length);
  reader_single_photon_numu.AddVariable("shw_sp_br3_1_sg_length",&tagger.shw_sp_br3_1_sg_length);
  reader_single_photon_numu.AddVariable("shw_sp_br3_1_total_main_length",&tagger.shw_sp_br3_1_total_main_length);
  reader_single_photon_numu.AddVariable("shw_sp_br3_1_total_length",&tagger.shw_sp_br3_1_total_length);
  reader_single_photon_numu.AddVariable("shw_sp_br3_2_n_ele",&tagger.shw_sp_br3_2_n_ele);
  reader_single_photon_numu.AddVariable("shw_sp_br3_2_n_other",&tagger.shw_sp_br3_2_n_other);
  reader_single_photon_numu.AddVariable("shw_sp_br3_2_other_fid",&tagger.shw_sp_br3_2_other_fid);
  reader_single_photon_numu.AddVariable("shw_sp_br3_4_acc_length",&tagger.shw_sp_br3_4_acc_length);
  reader_single_photon_numu.AddVariable("shw_sp_br3_4_total_length",&tagger.shw_sp_br3_4_total_length);
  reader_single_photon_numu.AddVariable("shw_sp_br3_8_max_dQ_dx",&tagger.shw_sp_br3_8_max_dQ_dx);
  reader_single_photon_numu.AddVariable("shw_sp_br3_8_n_main_segs",&tagger.shw_sp_br3_8_n_main_segs);
  reader_single_photon_numu.AddVariable("shw_sp_lem_shower_main_length",&tagger.shw_sp_lem_shower_main_length);
  reader_single_photon_numu.AddVariable("shw_sp_lem_n_3seg",&tagger.shw_sp_lem_n_3seg);
  reader_single_photon_numu.AddVariable("shw_sp_lem_e_charge",&tagger.shw_sp_lem_e_charge);
  reader_single_photon_numu.AddVariable("shw_sp_lem_e_dQdx",&tagger.shw_sp_lem_e_dQdx);
  reader_single_photon_numu.AddVariable("shw_sp_lem_shower_num_main_segs",&tagger.shw_sp_lem_shower_num_main_segs);
  reader_single_photon_numu.AddVariable("shw_sp_max_dQ_dx_sample",&tagger.shw_sp_max_dQ_dx_sample);
  reader_single_photon_numu.AddVariable("shw_sp_n_below_threshold",&tagger.shw_sp_n_below_threshold);
  reader_single_photon_numu.AddVariable("shw_sp_n_below_zero",&tagger.shw_sp_n_below_zero);
  reader_single_photon_numu.AddVariable("shw_sp_n_lowest",&tagger.shw_sp_n_lowest);
  reader_single_photon_numu.AddVariable("shw_sp_n_highest",&tagger.shw_sp_n_highest);
  reader_single_photon_numu.AddVariable("shw_sp_lowest_dQ_dx",&tagger.shw_sp_lowest_dQ_dx);
  reader_single_photon_numu.AddVariable("shw_sp_highest_dQ_dx",&tagger.shw_sp_highest_dQ_dx);
  reader_single_photon_numu.AddVariable("shw_sp_medium_dQ_dx",&tagger.shw_sp_medium_dQ_dx);
  reader_single_photon_numu.AddVariable("shw_sp_stem_length",&tagger.shw_sp_stem_length);
  reader_single_photon_numu.AddVariable("shw_sp_length_main",&tagger.shw_sp_length_main);
  reader_single_photon_numu.AddVariable("shw_sp_length_total",&tagger.shw_sp_length_total);
  reader_single_photon_numu.AddVariable("shw_sp_n_vertex",&tagger.shw_sp_n_vertex);
  reader_single_photon_numu.AddVariable("shw_sp_n_good_tracks",&tagger.shw_sp_n_good_tracks);
  reader_single_photon_numu.AddVariable("shw_sp_E_indirect_max_energy",&tagger.shw_sp_E_indirect_max_energy);
  reader_single_photon_numu.AddVariable("shw_sp_flag_all_above",&tagger.shw_sp_flag_all_above);
  reader_single_photon_numu.AddVariable("shw_sp_min_dQ_dx_5",&tagger.shw_sp_min_dQ_dx_5);
  reader_single_photon_numu.AddVariable("shw_sp_n_other_vertex",&tagger.shw_sp_n_other_vertex);
  reader_single_photon_numu.AddVariable("shw_sp_n_stem_size",&tagger.shw_sp_n_stem_size);
  reader_single_photon_numu.AddVariable("shw_sp_min_dis",&tagger.shw_sp_min_dis);
  reader_single_photon_numu.AddVariable("shw_sp_vec_mean_dedx",&tagger.shw_sp_vec_mean_dedx);
  reader_single_photon_numu.AddVariable("shw_sp_proton_length_1", &tagger.shw_sp_proton_length_1);
  reader_single_photon_numu.AddVariable("shw_sp_proton_dqdx_1", &tagger.shw_sp_proton_dqdx_1);
  reader_single_photon_numu.AddVariable("shw_sp_proton_energy_1",&tagger.shw_sp_proton_energy_1);
  reader_single_photon_numu.AddVariable("shw_sp_proton_length_2", &tagger.shw_sp_proton_length_2);
  reader_single_photon_numu.AddVariable("shw_sp_proton_dqdx_2", &tagger.shw_sp_proton_dqdx_2);
  reader_single_photon_numu.AddVariable("shw_sp_proton_energy_2",&tagger.shw_sp_proton_energy_2);
  reader_single_photon_numu.AddVariable("shw_sp_n_good_showers", &tagger.shw_sp_n_good_showers);
  reader_single_photon_numu.AddVariable("shw_sp_n_br1_showers", &tagger.shw_sp_n_br1_showers);
  reader_single_photon_numu.AddVariable("shw_sp_n_br2_showers", &tagger.shw_sp_n_br2_showers);
  reader_single_photon_numu.AddVariable("shw_sp_n_br3_showers", &tagger.shw_sp_n_br3_showers);
  reader_single_photon_numu.AddVariable("shw_sp_n_br4_showers", &tagger.shw_sp_n_br4_showers);
  reader_single_photon_numu.AddVariable("shw_sp_n_20br1_showers", &tagger.shw_sp_n_20br1_showers);
  reader_single_photon_numu.AddVariable("shw_sp_shw_vtx_dis", &tagger.shw_sp_shw_vtx_dis);
  reader_single_photon_numu.AddVariable("shw_sp_max_shw_dis",&tagger.shw_sp_max_shw_dis);
  reader_single_photon_numu.AddVariable("numu_1_score",&tagger.numu_1_score);
  reader_single_photon_numu.AddVariable("numu_score",&tagger.numu_score);

  reader_single_photon_numu.BookMVA( "MyBDT", "weights/single_photon_numu_bdt_final.xml");

  reader_single_photon_other.AddVariable("cosmict_flag_2",&tagger.cosmict_flag_2);
  reader_single_photon_other.AddVariable("cosmict_2_filled",&tagger.cosmict_2_filled);
  reader_single_photon_other.AddVariable("cosmict_2_particle_type",&tagger.cosmict_2_particle_type);
  reader_single_photon_other.AddVariable("cosmict_2_n_muon_tracks",&tagger.cosmict_2_n_muon_tracks);
  reader_single_photon_other.AddVariable("cosmict_2_total_shower_length",&tagger.cosmict_2_total_shower_length);
  reader_single_photon_other.AddVariable("cosmict_2_flag_inside",&tagger.cosmict_2_flag_inside);
  reader_single_photon_other.AddVariable("cosmict_2_angle_beam",&tagger.cosmict_2_angle_beam);
  reader_single_photon_other.AddVariable("cosmict_2_flag_dir_weak",&tagger.cosmict_2_flag_dir_weak);
  reader_single_photon_other.AddVariable("cosmict_2_dQ_dx_end",&tagger.cosmict_2_dQ_dx_end);
  reader_single_photon_other.AddVariable("cosmict_2_dQ_dx_front",&tagger.cosmict_2_dQ_dx_front);
  reader_single_photon_other.AddVariable("cosmict_2_theta",&tagger.cosmict_2_theta);
  reader_single_photon_other.AddVariable("cosmict_2_phi",&tagger.cosmict_2_phi);
  reader_single_photon_other.AddVariable("cosmict_2_valid_tracks",&tagger.cosmict_2_valid_tracks);
  reader_single_photon_other.AddVariable("cosmict_flag_4",&tagger.cosmict_flag_4);
  reader_single_photon_other.AddVariable("cosmict_4_filled",&tagger.cosmict_4_filled);
  reader_single_photon_other.AddVariable("cosmict_4_flag_inside",&tagger.cosmict_4_flag_inside);
  reader_single_photon_other.AddVariable("cosmict_4_angle_beam",&tagger.cosmict_4_angle_beam);
  reader_single_photon_other.AddVariable("cosmict_4_connected_showers",&tagger.cosmict_4_connected_showers);
  reader_single_photon_other.AddVariable("cosmict_flag_3",&tagger.cosmict_flag_3);
  reader_single_photon_other.AddVariable("cosmict_3_filled",&tagger.cosmict_3_filled);
  reader_single_photon_other.AddVariable("cosmict_3_flag_inside",&tagger.cosmict_3_flag_inside);
  reader_single_photon_other.AddVariable("cosmict_3_angle_beam",&tagger.cosmict_3_angle_beam);
  reader_single_photon_other.AddVariable("cosmict_3_flag_dir_weak",&tagger.cosmict_3_flag_dir_weak);
  reader_single_photon_other.AddVariable("cosmict_3_dQ_dx_end",&tagger.cosmict_3_dQ_dx_end);
  reader_single_photon_other.AddVariable("cosmict_3_dQ_dx_front",&tagger.cosmict_3_dQ_dx_front);
  reader_single_photon_other.AddVariable("cosmict_3_theta",&tagger.cosmict_3_theta);
  reader_single_photon_other.AddVariable("cosmict_3_phi",&tagger.cosmict_3_phi);
  reader_single_photon_other.AddVariable("cosmict_3_valid_tracks",&tagger.cosmict_3_valid_tracks);
  reader_single_photon_other.AddVariable("cosmict_flag_5",&tagger.cosmict_flag_5);
  reader_single_photon_other.AddVariable("cosmict_5_filled",&tagger.cosmict_5_filled);
  reader_single_photon_other.AddVariable("cosmict_5_flag_inside",&tagger.cosmict_5_flag_inside);
  reader_single_photon_other.AddVariable("cosmict_5_angle_beam",&tagger.cosmict_5_angle_beam);
  reader_single_photon_other.AddVariable("cosmict_5_connected_showers",&tagger.cosmict_5_connected_showers);
  reader_single_photon_other.AddVariable("cosmict_flag_6",&tagger.cosmict_flag_6);
  reader_single_photon_other.AddVariable("cosmict_6_filled",&tagger.cosmict_6_filled);
  reader_single_photon_other.AddVariable("cosmict_6_flag_dir_weak",&tagger.cosmict_6_flag_dir_weak);
  reader_single_photon_other.AddVariable("cosmict_6_flag_inside",&tagger.cosmict_6_flag_inside);
  reader_single_photon_other.AddVariable("cosmict_6_angle",&tagger.cosmict_6_angle);
  reader_single_photon_other.AddVariable("cosmict_flag_7",&tagger.cosmict_flag_7);
  reader_single_photon_other.AddVariable("cosmict_7_filled",&tagger.cosmict_7_filled);
  reader_single_photon_other.AddVariable("cosmict_7_flag_sec",&tagger.cosmict_7_flag_sec);
  reader_single_photon_other.AddVariable("cosmict_7_n_muon_tracks",&tagger.cosmict_7_n_muon_tracks);
  reader_single_photon_other.AddVariable("cosmict_7_total_shower_length",&tagger.cosmict_7_total_shower_length);
  reader_single_photon_other.AddVariable("cosmict_7_flag_inside",&tagger.cosmict_7_flag_inside);
  reader_single_photon_other.AddVariable("cosmict_7_angle_beam",&tagger.cosmict_7_angle_beam);
  reader_single_photon_other.AddVariable("cosmict_7_flag_dir_weak",&tagger.cosmict_7_flag_dir_weak);
  reader_single_photon_other.AddVariable("cosmict_7_dQ_dx_end",&tagger.cosmict_7_dQ_dx_end);
  reader_single_photon_other.AddVariable("cosmict_7_dQ_dx_front",&tagger.cosmict_7_dQ_dx_front);
  reader_single_photon_other.AddVariable("cosmict_7_theta",&tagger.cosmict_7_theta);
  reader_single_photon_other.AddVariable("cosmict_7_phi",&tagger.cosmict_7_phi);
  reader_single_photon_other.AddVariable("cosmict_flag_8",&tagger.cosmict_flag_8);
  reader_single_photon_other.AddVariable("cosmict_8_filled",&tagger.cosmict_8_filled);
  reader_single_photon_other.AddVariable("cosmict_8_flag_out",&tagger.cosmict_8_flag_out);
  reader_single_photon_other.AddVariable("cosmict_8_muon_length",&tagger.cosmict_8_muon_length);
  reader_single_photon_other.AddVariable("cosmict_8_acc_length",&tagger.cosmict_8_acc_length);
  reader_single_photon_other.AddVariable("cosmict_flag_9",&tagger.cosmict_flag_9);
  reader_single_photon_other.AddVariable("cosmic_flag",&tagger.cosmic_flag);
  reader_single_photon_other.AddVariable("cosmic_filled",&tagger.cosmic_filled);
  reader_single_photon_other.AddVariable("cosmict_flag",&tagger.cosmict_flag);
  reader_single_photon_other.AddVariable("numu_cc_flag",&tagger.numu_cc_flag);
  reader_single_photon_other.AddVariable("shw_sp_br2_num_valid_tracks", &tagger.shw_sp_br2_num_valid_tracks);
  reader_single_photon_other.AddVariable("shw_sp_br2_n_shower_main_segs",&tagger.shw_sp_br2_n_shower_main_segs);
  reader_single_photon_other.AddVariable("shw_sp_br2_sg_length",&tagger.shw_sp_br2_sg_length);
  reader_single_photon_other.AddVariable("shw_sp_br4_1_shower_main_length",&tagger.shw_sp_br4_1_shower_main_length);
  reader_single_photon_other.AddVariable("shw_sp_br4_1_shower_total_length",&tagger.shw_sp_br4_1_shower_total_length);
  reader_single_photon_other.AddVariable("shw_sp_br4_1_min_dis",&tagger.shw_sp_br4_1_min_dis);
  reader_single_photon_other.AddVariable("shw_sp_br4_1_flag_avoid_muon_check",&tagger.shw_sp_br4_1_flag_avoid_muon_check);
  reader_single_photon_other.AddVariable("shw_sp_br4_1_n_vtx_segs",&tagger.shw_sp_br4_1_n_vtx_segs);
  reader_single_photon_other.AddVariable("shw_sp_br4_1_n_main_segs",&tagger.shw_sp_br4_1_n_main_segs);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio_45",&tagger.shw_sp_br4_2_ratio_45);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio_35",&tagger.shw_sp_br4_2_ratio_35);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio_25",&tagger.shw_sp_br4_2_ratio_25);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio_15",&tagger.shw_sp_br4_2_ratio_15);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio1_45",&tagger.shw_sp_br4_2_ratio1_45);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio1_35",&tagger.shw_sp_br4_2_ratio1_35);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio1_25",&tagger.shw_sp_br4_2_ratio1_25);
  reader_single_photon_other.AddVariable("shw_sp_br4_2_ratio1_15",&tagger.shw_sp_br4_2_ratio1_15);
  reader_single_photon_other.AddVariable("shw_sp_lem_shower_main_length",&tagger.shw_sp_lem_shower_main_length);
  reader_single_photon_other.AddVariable("shw_sp_lem_n_3seg",&tagger.shw_sp_lem_n_3seg);
  reader_single_photon_other.AddVariable("shw_sp_lem_e_charge",&tagger.shw_sp_lem_e_charge);
  reader_single_photon_other.AddVariable("shw_sp_lem_e_dQdx",&tagger.shw_sp_lem_e_dQdx);
  reader_single_photon_other.AddVariable("shw_sp_lem_shower_num_main_segs",&tagger.shw_sp_lem_shower_num_main_segs);
  reader_single_photon_other.AddVariable("shw_sp_max_dQ_dx_sample",&tagger.shw_sp_max_dQ_dx_sample);
  reader_single_photon_other.AddVariable("shw_sp_n_below_threshold",&tagger.shw_sp_n_below_threshold);
  reader_single_photon_other.AddVariable("shw_sp_n_below_zero",&tagger.shw_sp_n_below_zero);
  reader_single_photon_other.AddVariable("shw_sp_n_lowest",&tagger.shw_sp_n_lowest);
  reader_single_photon_other.AddVariable("shw_sp_n_highest",&tagger.shw_sp_n_highest);
  reader_single_photon_other.AddVariable("shw_sp_lowest_dQ_dx",&tagger.shw_sp_lowest_dQ_dx);
  reader_single_photon_other.AddVariable("shw_sp_highest_dQ_dx",&tagger.shw_sp_highest_dQ_dx);
  reader_single_photon_other.AddVariable("shw_sp_medium_dQ_dx",&tagger.shw_sp_medium_dQ_dx);
  reader_single_photon_other.AddVariable("shw_sp_stem_length",&tagger.shw_sp_stem_length);
  reader_single_photon_other.AddVariable("shw_sp_length_main",&tagger.shw_sp_length_main);
  reader_single_photon_other.AddVariable("shw_sp_length_total",&tagger.shw_sp_length_total);
  reader_single_photon_other.AddVariable("shw_sp_n_vertex",&tagger.shw_sp_n_vertex);
  reader_single_photon_other.AddVariable("shw_sp_n_good_tracks",&tagger.shw_sp_n_good_tracks);
  reader_single_photon_other.AddVariable("shw_sp_E_indirect_max_energy",&tagger.shw_sp_E_indirect_max_energy);
  reader_single_photon_other.AddVariable("shw_sp_flag_all_above",&tagger.shw_sp_flag_all_above);
  reader_single_photon_other.AddVariable("shw_sp_min_dQ_dx_5",&tagger.shw_sp_min_dQ_dx_5);
  reader_single_photon_other.AddVariable("shw_sp_n_other_vertex",&tagger.shw_sp_n_other_vertex);
  reader_single_photon_other.AddVariable("shw_sp_n_stem_size",&tagger.shw_sp_n_stem_size);
  reader_single_photon_other.AddVariable("shw_sp_min_dis",&tagger.shw_sp_min_dis);
  reader_single_photon_other.AddVariable("shw_sp_vec_mean_dedx",&tagger.shw_sp_vec_mean_dedx);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_0",&tagger.shw_sp_vec_dQ_dx_0);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_1",&tagger.shw_sp_vec_dQ_dx_1);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_2",&tagger.shw_sp_vec_dQ_dx_2);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_3",&tagger.shw_sp_vec_dQ_dx_3);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_4",&tagger.shw_sp_vec_dQ_dx_4);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_5",&tagger.shw_sp_vec_dQ_dx_5);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_6",&tagger.shw_sp_vec_dQ_dx_6);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_7",&tagger.shw_sp_vec_dQ_dx_7);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_8",&tagger.shw_sp_vec_dQ_dx_8);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_9",&tagger.shw_sp_vec_dQ_dx_9);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_10",&tagger.shw_sp_vec_dQ_dx_10);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_11",&tagger.shw_sp_vec_dQ_dx_11);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_12",&tagger.shw_sp_vec_dQ_dx_12);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_13",&tagger.shw_sp_vec_dQ_dx_13);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_14",&tagger.shw_sp_vec_dQ_dx_14);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_15",&tagger.shw_sp_vec_dQ_dx_15);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_16",&tagger.shw_sp_vec_dQ_dx_16);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_17",&tagger.shw_sp_vec_dQ_dx_17);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_18",&tagger.shw_sp_vec_dQ_dx_18);
  reader_single_photon_other.AddVariable("shw_sp_vec_dQ_dx_19",&tagger.shw_sp_vec_dQ_dx_19);
  reader_single_photon_other.AddVariable("shw_sp_vec_median_dedx",&tagger.shw_sp_vec_median_dedx);
  reader_single_photon_other.AddVariable("shw_sp_proton_length_1", &tagger.shw_sp_proton_length_1);
  reader_single_photon_other.AddVariable("shw_sp_proton_dqdx_1", &tagger.shw_sp_proton_dqdx_1);
  reader_single_photon_other.AddVariable("shw_sp_proton_energy_1",&tagger.shw_sp_proton_energy_1);
  reader_single_photon_other.AddVariable("shw_sp_proton_length_2", &tagger.shw_sp_proton_length_2);
  reader_single_photon_other.AddVariable("shw_sp_proton_dqdx_2", &tagger.shw_sp_proton_dqdx_2);
  reader_single_photon_other.AddVariable("shw_sp_proton_energy_2",&tagger.shw_sp_proton_energy_2);
  reader_single_photon_other.AddVariable("shw_sp_n_good_showers", &tagger.shw_sp_n_good_showers);
  reader_single_photon_other.AddVariable("shw_sp_n_br1_showers", &tagger.shw_sp_n_br1_showers);
  reader_single_photon_other.AddVariable("shw_sp_n_br2_showers", &tagger.shw_sp_n_br2_showers);
  reader_single_photon_other.AddVariable("shw_sp_n_br3_showers", &tagger.shw_sp_n_br3_showers);
  reader_single_photon_other.AddVariable("shw_sp_n_br4_showers", &tagger.shw_sp_n_br4_showers);
  reader_single_photon_other.AddVariable("shw_sp_n_20br1_showers", &tagger.shw_sp_n_20br1_showers);
  reader_single_photon_other.AddVariable("shw_sp_shw_vtx_dis", &tagger.shw_sp_shw_vtx_dis);
  reader_single_photon_other.AddVariable("shw_sp_max_shw_dis",&tagger.shw_sp_max_shw_dis);
  reader_single_photon_other.AddVariable("shw_sp_num_mip_tracks",&tagger.shw_sp_num_mip_tracks);
  reader_single_photon_other.AddVariable("shw_sp_num_muons",&tagger.shw_sp_num_muons);
  reader_single_photon_other.AddVariable("shw_sp_num_protons",&tagger.shw_sp_num_protons);
  reader_single_photon_other.AddVariable("shw_sp_hol_2_min_angle",&tagger.shw_sp_hol_2_min_angle);
  reader_single_photon_other.AddVariable("cosmict_10_score",&tagger.cosmict_10_score);
  reader_single_photon_other.AddVariable("numu_1_score",&tagger.numu_1_score);
  reader_single_photon_other.AddVariable("numu_2_score",&tagger.numu_2_score);
  reader_single_photon_other.AddVariable("numu_score",&tagger.numu_score);
  reader_single_photon_other.AddVariable("reco_nuvtxY",&pfeval.reco_nuvtxY);

  reader_single_photon_other.BookMVA( "MyBDT", "weights/single_photon_other_bdt_final.xml");

  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_1_mass",&tagger.shw_sp_pio_1_mass);
  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_1_pio_type",&tagger.shw_sp_pio_1_pio_type);
  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_1_energy_1",&tagger.shw_sp_pio_1_energy_1);
  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_1_energy_2",&tagger.shw_sp_pio_1_energy_2);
  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_1_dis_1",&tagger.shw_sp_pio_1_dis_1);
  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_1_dis_2",&tagger.shw_sp_pio_1_dis_2);
  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_mip_id",&tagger.shw_sp_pio_mip_id);
  reader_single_photon_ncpi0.AddVariable("shw_sp_pio_flag_pio",&tagger.shw_sp_pio_flag_pio);
  reader_single_photon_ncpi0.AddVariable("shw_sp_n_20br1_showers",&tagger.shw_sp_n_20br1_showers);
  reader_single_photon_ncpi0.AddVariable("shw_sp_n_br1_showers",&tagger.shw_sp_n_br1_showers);
  reader_single_photon_ncpi0.AddVariable("shw_sp_n_br2_showers",&tagger.shw_sp_n_br2_showers);
  reader_single_photon_ncpi0.AddVariable("shw_sp_n_br3_showers",&tagger.shw_sp_n_br3_showers);
  reader_single_photon_ncpi0.AddVariable("shw_sp_n_br4_showers",&tagger.shw_sp_n_br4_showers);
  reader_single_photon_ncpi0.AddVariable("shw_sp_E_indirect_max_energy",&tagger.shw_sp_E_indirect_max_energy);
  reader_single_photon_ncpi0.AddVariable("shw_sp_lol_3_vtx_n_segs",&tagger.shw_sp_lol_3_vtx_n_segs);
  reader_single_photon_ncpi0.AddVariable("shw_sp_hol_1_min_length",&tagger.shw_sp_hol_1_min_length);
  reader_single_photon_ncpi0.AddVariable("shw_sp_hol_2_min_angle",&tagger.shw_sp_hol_2_min_angle);
  reader_single_photon_ncpi0.AddVariable("shw_sp_hol_1_min_angle",&tagger.shw_sp_hol_1_min_angle);
  reader_single_photon_ncpi0.AddVariable("tro_5_score",&tagger.tro_5_score);
  reader_single_photon_ncpi0.AddVariable("tro_4_score",&tagger.tro_4_score);
  reader_single_photon_ncpi0.AddVariable("tro_2_score",&tagger.tro_2_score);
  reader_single_photon_ncpi0.AddVariable("tro_1_score",&tagger.tro_1_score);
  reader_single_photon_ncpi0.AddVariable("stw_4_score",&tagger.stw_4_score);
  reader_single_photon_ncpi0.AddVariable("stw_3_score",&tagger.stw_3_score);
  reader_single_photon_ncpi0.AddVariable("stw_2_score",&tagger.stw_2_score);
  reader_single_photon_ncpi0.AddVariable("sig_2_score",&tagger.sig_2_score);
  reader_single_photon_ncpi0.AddVariable("sig_1_score",&tagger.sig_1_score);
  reader_single_photon_ncpi0.AddVariable("pio_2_score",&tagger.pio_2_score);
  reader_single_photon_ncpi0.AddVariable("lol_2_score",&tagger.lol_2_score);
  reader_single_photon_ncpi0.AddVariable("lol_1_score",&tagger.lol_1_score);
  reader_single_photon_ncpi0.AddVariable("br3_6_score",&tagger.br3_6_score);
  reader_single_photon_ncpi0.AddVariable("br3_5_score",&tagger.br3_5_score);
  reader_single_photon_ncpi0.AddVariable("br3_3_score",&tagger.br3_3_score);
  reader_single_photon_ncpi0.AddVariable("kine_pio_mass",&kine.kine_pio_mass);
  reader_single_photon_ncpi0.AddVariable("kine_pio_flag",&temp_kine_pio_flag);
  //reader_single_photon_ncpi0.AddVariable("kine_pio_flag",&kine.kine_pio_flag);
  reader_single_photon_ncpi0.AddVariable("kine_pio_vtx_dis",&kine.kine_pio_vtx_dis);
  reader_single_photon_ncpi0.AddVariable("kine_pio_energy_1",&kine.kine_pio_energy_1);
  reader_single_photon_ncpi0.AddVariable("kine_pio_theta_1",&kine.kine_pio_theta_1);
  reader_single_photon_ncpi0.AddVariable("kine_pio_phi_1",&kine.kine_pio_phi_1);
  reader_single_photon_ncpi0.AddVariable("kine_pio_dis_1",&kine.kine_pio_dis_1);
  reader_single_photon_ncpi0.AddVariable("kine_pio_energy_2",&kine.kine_pio_energy_2);
  reader_single_photon_ncpi0.AddVariable("kine_pio_theta_2",&kine.kine_pio_theta_2);
  reader_single_photon_ncpi0.AddVariable("kine_pio_phi_2",&kine.kine_pio_phi_2);
  reader_single_photon_ncpi0.AddVariable("kine_pio_dis_2",&kine.kine_pio_dis_2);
  reader_single_photon_ncpi0.AddVariable("kine_pio_angle",&kine.kine_pio_angle);

  reader_single_photon_ncpi0.BookMVA( "MyBDT", "weights/single_photon_ncpi0_bdt_final.xml");

  reader_single_photon_nue.AddVariable("shw_sp_max_dQ_dx_sample",&tagger.shw_sp_max_dQ_dx_sample);
  reader_single_photon_nue.AddVariable("shw_sp_n_below_threshold",&tagger.shw_sp_n_below_threshold);
  reader_single_photon_nue.AddVariable("shw_sp_n_below_zero",&tagger.shw_sp_n_below_zero);
  reader_single_photon_nue.AddVariable("shw_sp_n_lowest",&tagger.shw_sp_n_lowest);
  reader_single_photon_nue.AddVariable("shw_sp_n_highest",&tagger.shw_sp_n_highest);
  reader_single_photon_nue.AddVariable("shw_sp_lowest_dQ_dx",&tagger.shw_sp_lowest_dQ_dx);
  reader_single_photon_nue.AddVariable("shw_sp_highest_dQ_dx",&tagger.shw_sp_highest_dQ_dx);
  reader_single_photon_nue.AddVariable("shw_sp_medium_dQ_dx",&tagger.shw_sp_medium_dQ_dx);
  reader_single_photon_nue.AddVariable("shw_sp_stem_length",&tagger.shw_sp_stem_length);
  reader_single_photon_nue.AddVariable("shw_sp_length_main",&tagger.shw_sp_length_main);
  reader_single_photon_nue.AddVariable("shw_sp_length_total",&tagger.shw_sp_length_total);
  reader_single_photon_nue.AddVariable("shw_sp_n_vertex",&tagger.shw_sp_n_vertex);
  reader_single_photon_nue.AddVariable("shw_sp_n_good_tracks",&tagger.shw_sp_n_good_tracks);
  reader_single_photon_nue.AddVariable("shw_sp_E_indirect_max_energy",&tagger.shw_sp_E_indirect_max_energy);
  reader_single_photon_nue.AddVariable("shw_sp_flag_all_above",&tagger.shw_sp_flag_all_above);
  reader_single_photon_nue.AddVariable("shw_sp_min_dQ_dx_5",&tagger.shw_sp_min_dQ_dx_5);
  reader_single_photon_nue.AddVariable("shw_sp_n_other_vertex",&tagger.shw_sp_n_other_vertex);
  reader_single_photon_nue.AddVariable("shw_sp_n_stem_size",&tagger.shw_sp_n_stem_size);
  reader_single_photon_nue.AddVariable("shw_sp_min_dis",&tagger.shw_sp_min_dis);
  reader_single_photon_nue.AddVariable("shw_sp_vec_mean_dedx",&tagger.shw_sp_vec_mean_dedx);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_0",&tagger.shw_sp_vec_dQ_dx_0);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_1",&tagger.shw_sp_vec_dQ_dx_1);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_2",&tagger.shw_sp_vec_dQ_dx_2);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_3",&tagger.shw_sp_vec_dQ_dx_3);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_4",&tagger.shw_sp_vec_dQ_dx_4);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_5",&tagger.shw_sp_vec_dQ_dx_5);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_6",&tagger.shw_sp_vec_dQ_dx_6);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_7",&tagger.shw_sp_vec_dQ_dx_7);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_8",&tagger.shw_sp_vec_dQ_dx_8);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_9",&tagger.shw_sp_vec_dQ_dx_9);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_10",&tagger.shw_sp_vec_dQ_dx_10);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_11",&tagger.shw_sp_vec_dQ_dx_11);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_12",&tagger.shw_sp_vec_dQ_dx_12);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_13",&tagger.shw_sp_vec_dQ_dx_13);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_14",&tagger.shw_sp_vec_dQ_dx_14);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_15",&tagger.shw_sp_vec_dQ_dx_15);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_16",&tagger.shw_sp_vec_dQ_dx_16);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_17",&tagger.shw_sp_vec_dQ_dx_17);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_18",&tagger.shw_sp_vec_dQ_dx_18);
  reader_single_photon_nue.AddVariable("shw_sp_vec_dQ_dx_19",&tagger.shw_sp_vec_dQ_dx_19);
  reader_single_photon_nue.AddVariable("shw_sp_vec_median_dedx",&tagger.shw_sp_vec_median_dedx);
  reader_single_photon_nue.AddVariable("shw_sp_proton_length_1", &tagger.shw_sp_proton_length_1);
  reader_single_photon_nue.AddVariable("shw_sp_proton_dqdx_1", &tagger.shw_sp_proton_dqdx_1);
  reader_single_photon_nue.AddVariable("shw_sp_proton_energy_1",&tagger.shw_sp_proton_energy_1);
  reader_single_photon_nue.AddVariable("shw_sp_proton_length_2", &tagger.shw_sp_proton_length_2);
  reader_single_photon_nue.AddVariable("shw_sp_proton_dqdx_2", &tagger.shw_sp_proton_dqdx_2);
  reader_single_photon_nue.AddVariable("shw_sp_proton_energy_2",&tagger.shw_sp_proton_energy_2);
  reader_single_photon_nue.AddVariable("shw_sp_n_good_showers", &tagger.shw_sp_n_good_showers);
  reader_single_photon_nue.AddVariable("shw_sp_n_20mev_showers", &tagger.shw_sp_n_20mev_showers);
  reader_single_photon_nue.AddVariable("shw_sp_n_br1_showers", &tagger.shw_sp_n_br1_showers);
  reader_single_photon_nue.AddVariable("shw_sp_n_br2_showers", &tagger.shw_sp_n_br2_showers);
  reader_single_photon_nue.AddVariable("shw_sp_n_br3_showers", &tagger.shw_sp_n_br3_showers);
  reader_single_photon_nue.AddVariable("shw_sp_n_br4_showers", &tagger.shw_sp_n_br4_showers);
  reader_single_photon_nue.AddVariable("shw_sp_n_20br1_showers", &tagger.shw_sp_n_20br1_showers);
  reader_single_photon_nue.AddVariable("shw_sp_shw_vtx_dis", &tagger.shw_sp_shw_vtx_dis);
  reader_single_photon_nue.AddVariable("shw_sp_max_shw_dis",&tagger.shw_sp_max_shw_dis);

  reader_single_photon_nue.BookMVA( "MyBDT", "weights/single_photon_nue_bdt_final.xml");
  //

  TMVA::Reader reader_kdar_lowE;
  TMVA::Reader reader_kdar_hiE;

  reader_kdar_lowE.AddVariable("ssm_Nsm", &tagger.ssm_Nsm);
  reader_kdar_hiE.AddVariable("ssm_Nsm", &tagger.ssm_Nsm);
  reader_kdar_lowE.AddVariable("ssm_Nsm_wivtx", &tagger.ssm_Nsm_wivtx);
  reader_kdar_hiE.AddVariable("ssm_Nsm_wivtx", &tagger.ssm_Nsm_wivtx);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_fwd_1", &tagger.ssm_dq_dx_fwd_1);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_fwd_1", &tagger.ssm_dq_dx_fwd_1);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_fwd_2", &tagger.ssm_dq_dx_fwd_2);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_fwd_2", &tagger.ssm_dq_dx_fwd_2);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_fwd_3", &tagger.ssm_dq_dx_fwd_3);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_fwd_3", &tagger.ssm_dq_dx_fwd_3);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_fwd_4", &tagger.ssm_dq_dx_fwd_4);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_fwd_4", &tagger.ssm_dq_dx_fwd_4);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_fwd_5", &tagger.ssm_dq_dx_fwd_5);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_fwd_5", &tagger.ssm_dq_dx_fwd_5);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_bck_1", &tagger.ssm_dq_dx_bck_1);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_bck_1", &tagger.ssm_dq_dx_bck_1);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_bck_2", &tagger.ssm_dq_dx_bck_2);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_bck_2", &tagger.ssm_dq_dx_bck_2);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_bck_3", &tagger.ssm_dq_dx_bck_3);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_bck_3", &tagger.ssm_dq_dx_bck_3);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_bck_4", &tagger.ssm_dq_dx_bck_4);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_bck_4", &tagger.ssm_dq_dx_bck_4);
  reader_kdar_lowE.AddVariable("ssm_dq_dx_bck_5", &tagger.ssm_dq_dx_bck_5);
  reader_kdar_hiE.AddVariable("ssm_dq_dx_bck_5", &tagger.ssm_dq_dx_bck_5);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_fwd_12", &tagger.ssm_d_dq_dx_fwd_12);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_fwd_12", &tagger.ssm_d_dq_dx_fwd_12);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_fwd_23", &tagger.ssm_d_dq_dx_fwd_23);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_fwd_23", &tagger.ssm_d_dq_dx_fwd_23);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_fwd_34", &tagger.ssm_d_dq_dx_fwd_34);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_fwd_34", &tagger.ssm_d_dq_dx_fwd_34);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_fwd_45", &tagger.ssm_d_dq_dx_fwd_45);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_fwd_45", &tagger.ssm_d_dq_dx_fwd_45);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_bck_12", &tagger.ssm_d_dq_dx_bck_12);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_bck_12", &tagger.ssm_d_dq_dx_bck_12);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_bck_23", &tagger.ssm_d_dq_dx_bck_23);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_bck_23", &tagger.ssm_d_dq_dx_bck_23);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_bck_34", &tagger.ssm_d_dq_dx_bck_34);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_bck_34", &tagger.ssm_d_dq_dx_bck_34);
  reader_kdar_lowE.AddVariable("ssm_d_dq_dx_bck_45", &tagger.ssm_d_dq_dx_bck_45);
  reader_kdar_hiE.AddVariable("ssm_d_dq_dx_bck_45", &tagger.ssm_d_dq_dx_bck_45);
  reader_kdar_lowE.AddVariable("ssm_max_dq_dx_fwd_3", &tagger.ssm_max_dq_dx_fwd_3);
  reader_kdar_hiE.AddVariable("ssm_max_dq_dx_fwd_3", &tagger.ssm_max_dq_dx_fwd_3);
  reader_kdar_lowE.AddVariable("ssm_max_dq_dx_fwd_5", &tagger.ssm_max_dq_dx_fwd_5);
  reader_kdar_hiE.AddVariable("ssm_max_dq_dx_fwd_5", &tagger.ssm_max_dq_dx_fwd_5);
  reader_kdar_lowE.AddVariable("ssm_max_dq_dx_bck_3", &tagger.ssm_max_dq_dx_bck_3);
  reader_kdar_hiE.AddVariable("ssm_max_dq_dx_bck_3", &tagger.ssm_max_dq_dx_bck_3);
  reader_kdar_lowE.AddVariable("ssm_max_dq_dx_bck_5", &tagger.ssm_max_dq_dx_bck_5);
  reader_kdar_hiE.AddVariable("ssm_max_dq_dx_bck_5", &tagger.ssm_max_dq_dx_bck_5);
  reader_kdar_lowE.AddVariable("ssm_max_d_dq_dx_fwd_3", &tagger.ssm_max_d_dq_dx_fwd_3);
  reader_kdar_hiE.AddVariable("ssm_max_d_dq_dx_fwd_3", &tagger.ssm_max_d_dq_dx_fwd_3);
  reader_kdar_lowE.AddVariable("ssm_max_d_dq_dx_fwd_5", &tagger.ssm_max_d_dq_dx_fwd_5);
  reader_kdar_hiE.AddVariable("ssm_max_d_dq_dx_fwd_5", &tagger.ssm_max_d_dq_dx_fwd_5);
  reader_kdar_lowE.AddVariable("ssm_max_d_dq_dx_bck_3", &tagger.ssm_max_d_dq_dx_bck_3);
  reader_kdar_hiE.AddVariable("ssm_max_d_dq_dx_bck_3", &tagger.ssm_max_d_dq_dx_bck_3);
  reader_kdar_lowE.AddVariable("ssm_max_d_dq_dx_bck_5", &tagger.ssm_max_d_dq_dx_bck_5);
  reader_kdar_hiE.AddVariable("ssm_max_d_dq_dx_bck_5", &tagger.ssm_max_d_dq_dx_bck_5);
  reader_kdar_lowE.AddVariable("ssm_medium_dq_dx", &tagger.ssm_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_medium_dq_dx", &tagger.ssm_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_medium_dq_dx_bp", &tagger.ssm_medium_dq_dx_bp);
  reader_kdar_hiE.AddVariable("ssm_medium_dq_dx_bp", &tagger.ssm_medium_dq_dx_bp);
  reader_kdar_lowE.AddVariable("ssm_vtx_activity", &tagger.ssm_vtx_activity);
  reader_kdar_hiE.AddVariable("ssm_vtx_activity", &tagger.ssm_vtx_activity);
  reader_kdar_lowE.AddVariable("ssm_pdg", &tagger.ssm_pdg);
  reader_kdar_hiE.AddVariable("ssm_pdg", &tagger.ssm_pdg);
  reader_kdar_lowE.AddVariable("ssm_score_mu_fwd", &tagger.ssm_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_score_mu_fwd", &tagger.ssm_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_score_p_fwd", &tagger.ssm_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_score_p_fwd", &tagger.ssm_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_score_e_fwd", &tagger.ssm_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_score_e_fwd", &tagger.ssm_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_score_mu_bck", &tagger.ssm_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_score_mu_bck", &tagger.ssm_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_score_p_bck", &tagger.ssm_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_score_p_bck", &tagger.ssm_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_score_e_bck", &tagger.ssm_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_score_e_bck", &tagger.ssm_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_score_mu_fwd_bp", &tagger.ssm_score_mu_fwd_bp);
  reader_kdar_hiE.AddVariable("ssm_score_mu_fwd_bp", &tagger.ssm_score_mu_fwd_bp);
  reader_kdar_lowE.AddVariable("ssm_score_p_fwd_bp", &tagger.ssm_score_p_fwd_bp);
  reader_kdar_hiE.AddVariable("ssm_score_p_fwd_bp", &tagger.ssm_score_p_fwd_bp);
  reader_kdar_lowE.AddVariable("ssm_score_e_fwd_bp", &tagger.ssm_score_e_fwd_bp);
  reader_kdar_hiE.AddVariable("ssm_score_e_fwd_bp", &tagger.ssm_score_e_fwd_bp);
  reader_kdar_lowE.AddVariable("ssm_length_ratio", &tagger.ssm_length_ratio);
  reader_kdar_hiE.AddVariable("ssm_length_ratio", &tagger.ssm_length_ratio);
  reader_kdar_lowE.AddVariable("ssm_n_prim_tracks_1", &tagger.ssm_n_prim_tracks_1);
  reader_kdar_hiE.AddVariable("ssm_n_prim_tracks_1", &tagger.ssm_n_prim_tracks_1);
  reader_kdar_lowE.AddVariable("ssm_n_prim_tracks_3", &tagger.ssm_n_prim_tracks_3);
  reader_kdar_hiE.AddVariable("ssm_n_prim_tracks_3", &tagger.ssm_n_prim_tracks_3);
  reader_kdar_lowE.AddVariable("ssm_n_prim_tracks_5", &tagger.ssm_n_prim_tracks_5);
  reader_kdar_hiE.AddVariable("ssm_n_prim_tracks_5", &tagger.ssm_n_prim_tracks_5);
  reader_kdar_lowE.AddVariable("ssm_n_prim_tracks_8", &tagger.ssm_n_prim_tracks_8);
  reader_kdar_hiE.AddVariable("ssm_n_prim_tracks_8", &tagger.ssm_n_prim_tracks_8);
  reader_kdar_lowE.AddVariable("ssm_n_prim_tracks_11", &tagger.ssm_n_prim_tracks_11);
  reader_kdar_hiE.AddVariable("ssm_n_prim_tracks_11", &tagger.ssm_n_prim_tracks_11);
  reader_kdar_lowE.AddVariable("ssm_n_all_tracks_1", &tagger.ssm_n_all_tracks_1);
  reader_kdar_hiE.AddVariable("ssm_n_all_tracks_1", &tagger.ssm_n_all_tracks_1);
  reader_kdar_lowE.AddVariable("ssm_n_all_tracks_3", &tagger.ssm_n_all_tracks_3);
  reader_kdar_hiE.AddVariable("ssm_n_all_tracks_3", &tagger.ssm_n_all_tracks_3);
  reader_kdar_lowE.AddVariable("ssm_n_all_tracks_5", &tagger.ssm_n_all_tracks_5);
  reader_kdar_hiE.AddVariable("ssm_n_all_tracks_5", &tagger.ssm_n_all_tracks_5);
  reader_kdar_lowE.AddVariable("ssm_n_all_tracks_8", &tagger.ssm_n_all_tracks_8);
  reader_kdar_hiE.AddVariable("ssm_n_all_tracks_8", &tagger.ssm_n_all_tracks_8);
  reader_kdar_lowE.AddVariable("ssm_n_all_tracks_11", &tagger.ssm_n_all_tracks_11);
  reader_kdar_hiE.AddVariable("ssm_n_all_tracks_11", &tagger.ssm_n_all_tracks_11);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_tracks_1", &tagger.ssm_n_daughter_tracks_1);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_tracks_1", &tagger.ssm_n_daughter_tracks_1);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_tracks_3", &tagger.ssm_n_daughter_tracks_3);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_tracks_3", &tagger.ssm_n_daughter_tracks_3);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_tracks_5", &tagger.ssm_n_daughter_tracks_5);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_tracks_5", &tagger.ssm_n_daughter_tracks_5);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_tracks_8", &tagger.ssm_n_daughter_tracks_8);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_tracks_8", &tagger.ssm_n_daughter_tracks_8);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_tracks_11", &tagger.ssm_n_daughter_tracks_11);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_tracks_11", &tagger.ssm_n_daughter_tracks_11);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_all_1", &tagger.ssm_n_daughter_all_1);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_all_1", &tagger.ssm_n_daughter_all_1);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_all_3", &tagger.ssm_n_daughter_all_3);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_all_3", &tagger.ssm_n_daughter_all_3);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_all_5", &tagger.ssm_n_daughter_all_5);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_all_5", &tagger.ssm_n_daughter_all_5);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_all_8", &tagger.ssm_n_daughter_all_8);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_all_8", &tagger.ssm_n_daughter_all_8);
  reader_kdar_lowE.AddVariable("ssm_n_daughter_all_11", &tagger.ssm_n_daughter_all_11);
  reader_kdar_hiE.AddVariable("ssm_n_daughter_all_11", &tagger.ssm_n_daughter_all_11);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_pdg", &tagger.ssm_prim_track1_pdg);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_pdg", &tagger.ssm_prim_track1_pdg);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_score_mu_fwd", &tagger.ssm_prim_track1_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_score_mu_fwd", &tagger.ssm_prim_track1_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_score_p_fwd", &tagger.ssm_prim_track1_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_score_p_fwd", &tagger.ssm_prim_track1_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_score_e_fwd", &tagger.ssm_prim_track1_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_score_e_fwd", &tagger.ssm_prim_track1_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_score_mu_bck", &tagger.ssm_prim_track1_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_score_mu_bck", &tagger.ssm_prim_track1_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_score_p_bck", &tagger.ssm_prim_track1_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_score_p_bck", &tagger.ssm_prim_track1_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_score_e_bck", &tagger.ssm_prim_track1_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_score_e_bck", &tagger.ssm_prim_track1_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_max_dev", &tagger.ssm_prim_track1_max_dev);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_max_dev", &tagger.ssm_prim_track1_max_dev);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_medium_dq_dx", &tagger.ssm_prim_track1_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_medium_dq_dx", &tagger.ssm_prim_track1_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_add_daught_track_counts_1", &tagger.ssm_prim_track1_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_add_daught_track_counts_1", &tagger.ssm_prim_track1_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_add_daught_all_counts_1", &tagger.ssm_prim_track1_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_add_daught_all_counts_1", &tagger.ssm_prim_track1_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_add_daught_track_counts_5", &tagger.ssm_prim_track1_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_add_daught_track_counts_5", &tagger.ssm_prim_track1_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_add_daught_all_counts_5", &tagger.ssm_prim_track1_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_add_daught_all_counts_5", &tagger.ssm_prim_track1_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_add_daught_track_counts_11", &tagger.ssm_prim_track1_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_add_daught_track_counts_11", &tagger.ssm_prim_track1_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_prim_track1_add_daught_all_counts_11", &tagger.ssm_prim_track1_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_track1_add_daught_all_counts_11", &tagger.ssm_prim_track1_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_pdg", &tagger.ssm_prim_track2_pdg);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_pdg", &tagger.ssm_prim_track2_pdg);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_score_mu_fwd", &tagger.ssm_prim_track2_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_score_mu_fwd", &tagger.ssm_prim_track2_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_score_p_fwd", &tagger.ssm_prim_track2_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_score_p_fwd", &tagger.ssm_prim_track2_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_score_e_fwd", &tagger.ssm_prim_track2_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_score_e_fwd", &tagger.ssm_prim_track2_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_score_mu_bck", &tagger.ssm_prim_track2_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_score_mu_bck", &tagger.ssm_prim_track2_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_score_p_bck", &tagger.ssm_prim_track2_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_score_p_bck", &tagger.ssm_prim_track2_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_score_e_bck", &tagger.ssm_prim_track2_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_score_e_bck", &tagger.ssm_prim_track2_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_max_dev", &tagger.ssm_prim_track2_max_dev);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_max_dev", &tagger.ssm_prim_track2_max_dev);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_medium_dq_dx", &tagger.ssm_prim_track2_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_medium_dq_dx", &tagger.ssm_prim_track2_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_add_daught_track_counts_1", &tagger.ssm_prim_track2_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_add_daught_track_counts_1", &tagger.ssm_prim_track2_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_add_daught_all_counts_1", &tagger.ssm_prim_track2_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_add_daught_all_counts_1", &tagger.ssm_prim_track2_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_add_daught_track_counts_5", &tagger.ssm_prim_track2_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_add_daught_track_counts_5", &tagger.ssm_prim_track2_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_add_daught_all_counts_5", &tagger.ssm_prim_track2_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_add_daught_all_counts_5", &tagger.ssm_prim_track2_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_add_daught_track_counts_11", &tagger.ssm_prim_track2_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_add_daught_track_counts_11", &tagger.ssm_prim_track2_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_prim_track2_add_daught_all_counts_11", &tagger.ssm_prim_track2_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_track2_add_daught_all_counts_11", &tagger.ssm_prim_track2_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_pdg", &tagger.ssm_daught_track1_pdg);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_pdg", &tagger.ssm_daught_track1_pdg);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_score_mu_fwd", &tagger.ssm_daught_track1_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_score_mu_fwd", &tagger.ssm_daught_track1_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_score_p_fwd", &tagger.ssm_daught_track1_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_score_p_fwd", &tagger.ssm_daught_track1_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_score_e_fwd", &tagger.ssm_daught_track1_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_score_e_fwd", &tagger.ssm_daught_track1_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_score_mu_bck", &tagger.ssm_daught_track1_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_score_mu_bck", &tagger.ssm_daught_track1_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_score_p_bck", &tagger.ssm_daught_track1_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_score_p_bck", &tagger.ssm_daught_track1_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_score_e_bck", &tagger.ssm_daught_track1_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_score_e_bck", &tagger.ssm_daught_track1_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_length", &tagger.ssm_daught_track1_length);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_length", &tagger.ssm_daught_track1_length);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_direct_length", &tagger.ssm_daught_track1_direct_length);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_direct_length", &tagger.ssm_daught_track1_direct_length);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_length_ratio", &tagger.ssm_daught_track1_length_ratio);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_length_ratio", &tagger.ssm_daught_track1_length_ratio);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_max_dev", &tagger.ssm_daught_track1_max_dev);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_max_dev", &tagger.ssm_daught_track1_max_dev);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_kine_energy_range", &tagger.ssm_daught_track1_kine_energy_range);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_kine_energy_range", &tagger.ssm_daught_track1_kine_energy_range);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_kine_energy_range_mu", &tagger.ssm_daught_track1_kine_energy_range_mu);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_kine_energy_range_mu", &tagger.ssm_daught_track1_kine_energy_range_mu);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_kine_energy_range_p", &tagger.ssm_daught_track1_kine_energy_range_p);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_kine_energy_range_p", &tagger.ssm_daught_track1_kine_energy_range_p);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_kine_energy_range_e", &tagger.ssm_daught_track1_kine_energy_range_e);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_kine_energy_range_e", &tagger.ssm_daught_track1_kine_energy_range_e);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_kine_energy_cal", &tagger.ssm_daught_track1_kine_energy_cal);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_kine_energy_cal", &tagger.ssm_daught_track1_kine_energy_cal);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_medium_dq_dx", &tagger.ssm_daught_track1_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_medium_dq_dx", &tagger.ssm_daught_track1_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_x_dir", &tagger.ssm_daught_track1_x_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_x_dir", &tagger.ssm_daught_track1_x_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_y_dir", &tagger.ssm_daught_track1_y_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_y_dir", &tagger.ssm_daught_track1_y_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_z_dir", &tagger.ssm_daught_track1_z_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_z_dir", &tagger.ssm_daught_track1_z_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_add_daught_track_counts_1", &tagger.ssm_daught_track1_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_add_daught_track_counts_1", &tagger.ssm_daught_track1_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_add_daught_all_counts_1", &tagger.ssm_daught_track1_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_add_daught_all_counts_1", &tagger.ssm_daught_track1_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_add_daught_track_counts_5", &tagger.ssm_daught_track1_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_add_daught_track_counts_5", &tagger.ssm_daught_track1_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_add_daught_all_counts_5", &tagger.ssm_daught_track1_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_add_daught_all_counts_5", &tagger.ssm_daught_track1_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_add_daught_track_counts_11", &tagger.ssm_daught_track1_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_add_daught_track_counts_11", &tagger.ssm_daught_track1_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_track1_add_daught_all_counts_11", &tagger.ssm_daught_track1_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_track1_add_daught_all_counts_11", &tagger.ssm_daught_track1_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_pdg", &tagger.ssm_daught_track2_pdg);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_pdg", &tagger.ssm_daught_track2_pdg);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_score_mu_fwd", &tagger.ssm_daught_track2_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_score_mu_fwd", &tagger.ssm_daught_track2_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_score_p_fwd", &tagger.ssm_daught_track2_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_score_p_fwd", &tagger.ssm_daught_track2_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_score_e_fwd", &tagger.ssm_daught_track2_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_score_e_fwd", &tagger.ssm_daught_track2_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_score_mu_bck", &tagger.ssm_daught_track2_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_score_mu_bck", &tagger.ssm_daught_track2_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_score_p_bck", &tagger.ssm_daught_track2_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_score_p_bck", &tagger.ssm_daught_track2_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_score_e_bck", &tagger.ssm_daught_track2_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_score_e_bck", &tagger.ssm_daught_track2_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_length", &tagger.ssm_daught_track2_length);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_length", &tagger.ssm_daught_track2_length);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_direct_length", &tagger.ssm_daught_track2_direct_length);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_direct_length", &tagger.ssm_daught_track2_direct_length);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_length_ratio", &tagger.ssm_daught_track2_length_ratio);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_length_ratio", &tagger.ssm_daught_track2_length_ratio);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_max_dev", &tagger.ssm_daught_track2_max_dev);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_max_dev", &tagger.ssm_daught_track2_max_dev);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_kine_energy_range", &tagger.ssm_daught_track2_kine_energy_range);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_kine_energy_range", &tagger.ssm_daught_track2_kine_energy_range);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_kine_energy_range_mu", &tagger.ssm_daught_track2_kine_energy_range_mu);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_kine_energy_range_mu", &tagger.ssm_daught_track2_kine_energy_range_mu);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_kine_energy_range_p", &tagger.ssm_daught_track2_kine_energy_range_p);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_kine_energy_range_p", &tagger.ssm_daught_track2_kine_energy_range_p);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_kine_energy_range_e", &tagger.ssm_daught_track2_kine_energy_range_e);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_kine_energy_range_e", &tagger.ssm_daught_track2_kine_energy_range_e);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_kine_energy_cal", &tagger.ssm_daught_track2_kine_energy_cal);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_kine_energy_cal", &tagger.ssm_daught_track2_kine_energy_cal);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_medium_dq_dx", &tagger.ssm_daught_track2_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_medium_dq_dx", &tagger.ssm_daught_track2_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_x_dir", &tagger.ssm_daught_track2_x_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_x_dir", &tagger.ssm_daught_track2_x_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_y_dir", &tagger.ssm_daught_track2_y_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_y_dir", &tagger.ssm_daught_track2_y_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_z_dir", &tagger.ssm_daught_track2_z_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_z_dir", &tagger.ssm_daught_track2_z_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_add_daught_track_counts_1", &tagger.ssm_daught_track2_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_add_daught_track_counts_1", &tagger.ssm_daught_track2_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_add_daught_all_counts_1", &tagger.ssm_daught_track2_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_add_daught_all_counts_1", &tagger.ssm_daught_track2_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_add_daught_track_counts_5", &tagger.ssm_daught_track2_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_add_daught_track_counts_5", &tagger.ssm_daught_track2_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_add_daught_all_counts_5", &tagger.ssm_daught_track2_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_add_daught_all_counts_5", &tagger.ssm_daught_track2_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_add_daught_track_counts_11", &tagger.ssm_daught_track2_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_add_daught_track_counts_11", &tagger.ssm_daught_track2_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_track2_add_daught_all_counts_11", &tagger.ssm_daught_track2_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_track2_add_daught_all_counts_11", &tagger.ssm_daught_track2_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_pdg", &tagger.ssm_prim_shw1_pdg);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_pdg", &tagger.ssm_prim_shw1_pdg);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_score_mu_fwd", &tagger.ssm_prim_shw1_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_score_mu_fwd", &tagger.ssm_prim_shw1_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_score_p_fwd", &tagger.ssm_prim_shw1_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_score_p_fwd", &tagger.ssm_prim_shw1_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_score_e_fwd", &tagger.ssm_prim_shw1_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_score_e_fwd", &tagger.ssm_prim_shw1_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_score_mu_bck", &tagger.ssm_prim_shw1_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_score_mu_bck", &tagger.ssm_prim_shw1_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_score_p_bck", &tagger.ssm_prim_shw1_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_score_p_bck", &tagger.ssm_prim_shw1_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_score_e_bck", &tagger.ssm_prim_shw1_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_score_e_bck", &tagger.ssm_prim_shw1_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_max_dev", &tagger.ssm_prim_shw1_max_dev);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_max_dev", &tagger.ssm_prim_shw1_max_dev);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_medium_dq_dx", &tagger.ssm_prim_shw1_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_medium_dq_dx", &tagger.ssm_prim_shw1_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_add_daught_track_counts_1", &tagger.ssm_prim_shw1_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_add_daught_track_counts_1", &tagger.ssm_prim_shw1_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_add_daught_all_counts_1", &tagger.ssm_prim_shw1_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_add_daught_all_counts_1", &tagger.ssm_prim_shw1_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_add_daught_track_counts_5", &tagger.ssm_prim_shw1_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_add_daught_track_counts_5", &tagger.ssm_prim_shw1_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_add_daught_all_counts_5", &tagger.ssm_prim_shw1_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_add_daught_all_counts_5", &tagger.ssm_prim_shw1_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_add_daught_track_counts_11", &tagger.ssm_prim_shw1_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_add_daught_track_counts_11", &tagger.ssm_prim_shw1_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_prim_shw1_add_daught_all_counts_11", &tagger.ssm_prim_shw1_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_shw1_add_daught_all_counts_11", &tagger.ssm_prim_shw1_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_pdg", &tagger.ssm_prim_shw2_pdg);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_pdg", &tagger.ssm_prim_shw2_pdg);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_score_mu_fwd", &tagger.ssm_prim_shw2_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_score_mu_fwd", &tagger.ssm_prim_shw2_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_score_p_fwd", &tagger.ssm_prim_shw2_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_score_p_fwd", &tagger.ssm_prim_shw2_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_score_e_fwd", &tagger.ssm_prim_shw2_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_score_e_fwd", &tagger.ssm_prim_shw2_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_score_mu_bck", &tagger.ssm_prim_shw2_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_score_mu_bck", &tagger.ssm_prim_shw2_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_score_p_bck", &tagger.ssm_prim_shw2_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_score_p_bck", &tagger.ssm_prim_shw2_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_score_e_bck", &tagger.ssm_prim_shw2_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_score_e_bck", &tagger.ssm_prim_shw2_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_max_dev", &tagger.ssm_prim_shw2_max_dev);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_max_dev", &tagger.ssm_prim_shw2_max_dev);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_medium_dq_dx", &tagger.ssm_prim_shw2_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_medium_dq_dx", &tagger.ssm_prim_shw2_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_add_daught_track_counts_1", &tagger.ssm_prim_shw2_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_add_daught_track_counts_1", &tagger.ssm_prim_shw2_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_add_daught_all_counts_1", &tagger.ssm_prim_shw2_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_add_daught_all_counts_1", &tagger.ssm_prim_shw2_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_add_daught_track_counts_5", &tagger.ssm_prim_shw2_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_add_daught_track_counts_5", &tagger.ssm_prim_shw2_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_add_daught_all_counts_5", &tagger.ssm_prim_shw2_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_add_daught_all_counts_5", &tagger.ssm_prim_shw2_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_add_daught_track_counts_11", &tagger.ssm_prim_shw2_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_add_daught_track_counts_11", &tagger.ssm_prim_shw2_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_prim_shw2_add_daught_all_counts_11", &tagger.ssm_prim_shw2_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_prim_shw2_add_daught_all_counts_11", &tagger.ssm_prim_shw2_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_pdg", &tagger.ssm_daught_shw1_pdg);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_pdg", &tagger.ssm_daught_shw1_pdg);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_score_mu_fwd", &tagger.ssm_daught_shw1_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_score_mu_fwd", &tagger.ssm_daught_shw1_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_score_p_fwd", &tagger.ssm_daught_shw1_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_score_p_fwd", &tagger.ssm_daught_shw1_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_score_e_fwd", &tagger.ssm_daught_shw1_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_score_e_fwd", &tagger.ssm_daught_shw1_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_score_mu_bck", &tagger.ssm_daught_shw1_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_score_mu_bck", &tagger.ssm_daught_shw1_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_score_p_bck", &tagger.ssm_daught_shw1_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_score_p_bck", &tagger.ssm_daught_shw1_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_score_e_bck", &tagger.ssm_daught_shw1_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_score_e_bck", &tagger.ssm_daught_shw1_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_length", &tagger.ssm_daught_shw1_length);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_length", &tagger.ssm_daught_shw1_length);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_direct_length", &tagger.ssm_daught_shw1_direct_length);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_direct_length", &tagger.ssm_daught_shw1_direct_length);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_length_ratio", &tagger.ssm_daught_shw1_length_ratio);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_length_ratio", &tagger.ssm_daught_shw1_length_ratio);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_max_dev", &tagger.ssm_daught_shw1_max_dev);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_max_dev", &tagger.ssm_daught_shw1_max_dev);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_kine_energy_range", &tagger.ssm_daught_shw1_kine_energy_range);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_kine_energy_range", &tagger.ssm_daught_shw1_kine_energy_range);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_kine_energy_range_mu", &tagger.ssm_daught_shw1_kine_energy_range_mu);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_kine_energy_range_mu", &tagger.ssm_daught_shw1_kine_energy_range_mu);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_kine_energy_range_p", &tagger.ssm_daught_shw1_kine_energy_range_p);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_kine_energy_range_p", &tagger.ssm_daught_shw1_kine_energy_range_p);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_kine_energy_range_e", &tagger.ssm_daught_shw1_kine_energy_range_e);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_kine_energy_range_e", &tagger.ssm_daught_shw1_kine_energy_range_e);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_kine_energy_cal", &tagger.ssm_daught_shw1_kine_energy_cal);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_kine_energy_cal", &tagger.ssm_daught_shw1_kine_energy_cal);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_kine_energy_best", &tagger.ssm_daught_shw1_kine_energy_best);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_kine_energy_best", &tagger.ssm_daught_shw1_kine_energy_best);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_medium_dq_dx", &tagger.ssm_daught_shw1_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_medium_dq_dx", &tagger.ssm_daught_shw1_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_x_dir", &tagger.ssm_daught_shw1_x_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_x_dir", &tagger.ssm_daught_shw1_x_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_y_dir", &tagger.ssm_daught_shw1_y_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_y_dir", &tagger.ssm_daught_shw1_y_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_z_dir", &tagger.ssm_daught_shw1_z_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_z_dir", &tagger.ssm_daught_shw1_z_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_add_daught_track_counts_1", &tagger.ssm_daught_shw1_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_add_daught_track_counts_1", &tagger.ssm_daught_shw1_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_add_daught_all_counts_1", &tagger.ssm_daught_shw1_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_add_daught_all_counts_1", &tagger.ssm_daught_shw1_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_add_daught_track_counts_5", &tagger.ssm_daught_shw1_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_add_daught_track_counts_5", &tagger.ssm_daught_shw1_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_add_daught_all_counts_5", &tagger.ssm_daught_shw1_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_add_daught_all_counts_5", &tagger.ssm_daught_shw1_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_add_daught_track_counts_11", &tagger.ssm_daught_shw1_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_add_daught_track_counts_11", &tagger.ssm_daught_shw1_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_shw1_add_daught_all_counts_11", &tagger.ssm_daught_shw1_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_shw1_add_daught_all_counts_11", &tagger.ssm_daught_shw1_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_pdg", &tagger.ssm_daught_shw2_pdg);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_pdg", &tagger.ssm_daught_shw2_pdg);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_score_mu_fwd", &tagger.ssm_daught_shw2_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_score_mu_fwd", &tagger.ssm_daught_shw2_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_score_p_fwd", &tagger.ssm_daught_shw2_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_score_p_fwd", &tagger.ssm_daught_shw2_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_score_e_fwd", &tagger.ssm_daught_shw2_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_score_e_fwd", &tagger.ssm_daught_shw2_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_score_mu_bck", &tagger.ssm_daught_shw2_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_score_mu_bck", &tagger.ssm_daught_shw2_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_score_p_bck", &tagger.ssm_daught_shw2_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_score_p_bck", &tagger.ssm_daught_shw2_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_score_e_bck", &tagger.ssm_daught_shw2_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_score_e_bck", &tagger.ssm_daught_shw2_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_length", &tagger.ssm_daught_shw2_length);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_length", &tagger.ssm_daught_shw2_length);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_direct_length", &tagger.ssm_daught_shw2_direct_length);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_direct_length", &tagger.ssm_daught_shw2_direct_length);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_length_ratio", &tagger.ssm_daught_shw2_length_ratio);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_length_ratio", &tagger.ssm_daught_shw2_length_ratio);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_max_dev", &tagger.ssm_daught_shw2_max_dev);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_max_dev", &tagger.ssm_daught_shw2_max_dev);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_kine_energy_range", &tagger.ssm_daught_shw2_kine_energy_range);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_kine_energy_range", &tagger.ssm_daught_shw2_kine_energy_range);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_kine_energy_range_mu", &tagger.ssm_daught_shw2_kine_energy_range_mu);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_kine_energy_range_mu", &tagger.ssm_daught_shw2_kine_energy_range_mu);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_kine_energy_range_p", &tagger.ssm_daught_shw2_kine_energy_range_p);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_kine_energy_range_p", &tagger.ssm_daught_shw2_kine_energy_range_p);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_kine_energy_range_e", &tagger.ssm_daught_shw2_kine_energy_range_e);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_kine_energy_range_e", &tagger.ssm_daught_shw2_kine_energy_range_e);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_kine_energy_cal", &tagger.ssm_daught_shw2_kine_energy_cal);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_kine_energy_cal", &tagger.ssm_daught_shw2_kine_energy_cal);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_kine_energy_best", &tagger.ssm_daught_shw2_kine_energy_best);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_kine_energy_best", &tagger.ssm_daught_shw2_kine_energy_best);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_medium_dq_dx", &tagger.ssm_daught_shw2_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_medium_dq_dx", &tagger.ssm_daught_shw2_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_x_dir", &tagger.ssm_daught_shw2_x_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_x_dir", &tagger.ssm_daught_shw2_x_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_y_dir", &tagger.ssm_daught_shw2_y_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_y_dir", &tagger.ssm_daught_shw2_y_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_z_dir", &tagger.ssm_daught_shw2_z_dir);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_z_dir", &tagger.ssm_daught_shw2_z_dir);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_add_daught_track_counts_1", &tagger.ssm_daught_shw2_add_daught_track_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_add_daught_track_counts_1", &tagger.ssm_daught_shw2_add_daught_track_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_add_daught_all_counts_1", &tagger.ssm_daught_shw2_add_daught_all_counts_1);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_add_daught_all_counts_1", &tagger.ssm_daught_shw2_add_daught_all_counts_1);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_add_daught_track_counts_5", &tagger.ssm_daught_shw2_add_daught_track_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_add_daught_track_counts_5", &tagger.ssm_daught_shw2_add_daught_track_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_add_daught_all_counts_5", &tagger.ssm_daught_shw2_add_daught_all_counts_5);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_add_daught_all_counts_5", &tagger.ssm_daught_shw2_add_daught_all_counts_5);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_add_daught_track_counts_11", &tagger.ssm_daught_shw2_add_daught_track_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_add_daught_track_counts_11", &tagger.ssm_daught_shw2_add_daught_track_counts_11);
  reader_kdar_lowE.AddVariable("ssm_daught_shw2_add_daught_all_counts_11", &tagger.ssm_daught_shw2_add_daught_all_counts_11);
  reader_kdar_hiE.AddVariable("ssm_daught_shw2_add_daught_all_counts_11", &tagger.ssm_daught_shw2_add_daught_all_counts_11);
  reader_kdar_lowE.AddVariable("ssm_nu_angle_z", &tagger.ssm_nu_angle_z);
  reader_kdar_hiE.AddVariable("ssm_nu_angle_z", &tagger.ssm_nu_angle_z);
  reader_kdar_lowE.AddVariable("ssm_nu_angle_target", &tagger.ssm_nu_angle_target);
  reader_kdar_hiE.AddVariable("ssm_nu_angle_target", &tagger.ssm_nu_angle_target);
  reader_kdar_lowE.AddVariable("ssm_nu_angle_absorber", &tagger.ssm_nu_angle_absorber);
  reader_kdar_hiE.AddVariable("ssm_nu_angle_absorber", &tagger.ssm_nu_angle_absorber);
  reader_kdar_lowE.AddVariable("ssm_nu_angle_vertical", &tagger.ssm_nu_angle_vertical);
  reader_kdar_hiE.AddVariable("ssm_nu_angle_vertical", &tagger.ssm_nu_angle_vertical);
  reader_kdar_lowE.AddVariable("ssm_prim_nu_angle_z", &tagger.ssm_prim_nu_angle_z);
  reader_kdar_hiE.AddVariable("ssm_prim_nu_angle_z", &tagger.ssm_prim_nu_angle_z);
  reader_kdar_lowE.AddVariable("ssm_prim_nu_angle_target", &tagger.ssm_prim_nu_angle_target);
  reader_kdar_hiE.AddVariable("ssm_prim_nu_angle_target", &tagger.ssm_prim_nu_angle_target);
  reader_kdar_lowE.AddVariable("ssm_prim_nu_angle_absorber", &tagger.ssm_prim_nu_angle_absorber);
  reader_kdar_hiE.AddVariable("ssm_prim_nu_angle_absorber", &tagger.ssm_prim_nu_angle_absorber);
  reader_kdar_lowE.AddVariable("ssm_prim_nu_angle_vertical", &tagger.ssm_prim_nu_angle_vertical);
  reader_kdar_hiE.AddVariable("ssm_prim_nu_angle_vertical", &tagger.ssm_prim_nu_angle_vertical);
  reader_kdar_lowE.AddVariable("ssm_con_nu_angle_z", &tagger.ssm_con_nu_angle_z);
  reader_kdar_hiE.AddVariable("ssm_con_nu_angle_z", &tagger.ssm_con_nu_angle_z);
  reader_kdar_lowE.AddVariable("ssm_con_nu_angle_target", &tagger.ssm_con_nu_angle_target);
  reader_kdar_hiE.AddVariable("ssm_con_nu_angle_target", &tagger.ssm_con_nu_angle_target);
  reader_kdar_lowE.AddVariable("ssm_con_nu_angle_absorber", &tagger.ssm_con_nu_angle_absorber);
  reader_kdar_hiE.AddVariable("ssm_con_nu_angle_absorber", &tagger.ssm_con_nu_angle_absorber);
  reader_kdar_lowE.AddVariable("ssm_con_nu_angle_vertical", &tagger.ssm_con_nu_angle_vertical);
  reader_kdar_hiE.AddVariable("ssm_con_nu_angle_vertical", &tagger.ssm_con_nu_angle_vertical);
  reader_kdar_lowE.AddVariable("ssm_track_angle_z", &tagger.ssm_track_angle_z);
  reader_kdar_hiE.AddVariable("ssm_track_angle_z", &tagger.ssm_track_angle_z);
  reader_kdar_lowE.AddVariable("ssm_track_angle_target", &tagger.ssm_track_angle_target);
  reader_kdar_hiE.AddVariable("ssm_track_angle_target", &tagger.ssm_track_angle_target);
  reader_kdar_lowE.AddVariable("ssm_track_angle_absorber", &tagger.ssm_track_angle_absorber);
  reader_kdar_hiE.AddVariable("ssm_track_angle_absorber", &tagger.ssm_track_angle_absorber);
  reader_kdar_lowE.AddVariable("ssm_track_angle_vertical", &tagger.ssm_track_angle_vertical);
  reader_kdar_hiE.AddVariable("ssm_track_angle_vertical", &tagger.ssm_track_angle_vertical);
  reader_kdar_lowE.AddVariable("ssm_offvtx_length", &tagger.ssm_offvtx_length);
  reader_kdar_hiE.AddVariable("ssm_offvtx_length", &tagger.ssm_offvtx_length);
  reader_kdar_lowE.AddVariable("ssm_offvtx_energy", &tagger.ssm_offvtx_energy);
  reader_kdar_hiE.AddVariable("ssm_offvtx_energy", &tagger.ssm_offvtx_energy);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_tracks_1", &tagger.ssm_n_offvtx_tracks_1);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_tracks_1", &tagger.ssm_n_offvtx_tracks_1);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_tracks_3", &tagger.ssm_n_offvtx_tracks_3);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_tracks_3", &tagger.ssm_n_offvtx_tracks_3);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_tracks_5", &tagger.ssm_n_offvtx_tracks_5);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_tracks_5", &tagger.ssm_n_offvtx_tracks_5);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_tracks_8", &tagger.ssm_n_offvtx_tracks_8);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_tracks_8", &tagger.ssm_n_offvtx_tracks_8);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_tracks_11", &tagger.ssm_n_offvtx_tracks_11);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_tracks_11", &tagger.ssm_n_offvtx_tracks_11);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_showers_1", &tagger.ssm_n_offvtx_showers_1);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_showers_1", &tagger.ssm_n_offvtx_showers_1);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_showers_3", &tagger.ssm_n_offvtx_showers_3);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_showers_3", &tagger.ssm_n_offvtx_showers_3);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_showers_5", &tagger.ssm_n_offvtx_showers_5);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_showers_5", &tagger.ssm_n_offvtx_showers_5);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_showers_8", &tagger.ssm_n_offvtx_showers_8);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_showers_8", &tagger.ssm_n_offvtx_showers_8);
  reader_kdar_lowE.AddVariable("ssm_n_offvtx_showers_11", &tagger.ssm_n_offvtx_showers_11);
  reader_kdar_hiE.AddVariable("ssm_n_offvtx_showers_11", &tagger.ssm_n_offvtx_showers_11);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_pdg", &tagger.ssm_offvtx_track1_pdg);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_pdg", &tagger.ssm_offvtx_track1_pdg);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_score_mu_fwd", &tagger.ssm_offvtx_track1_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_score_mu_fwd", &tagger.ssm_offvtx_track1_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_score_p_fwd", &tagger.ssm_offvtx_track1_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_score_p_fwd", &tagger.ssm_offvtx_track1_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_score_e_fwd", &tagger.ssm_offvtx_track1_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_score_e_fwd", &tagger.ssm_offvtx_track1_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_score_mu_bck", &tagger.ssm_offvtx_track1_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_score_mu_bck", &tagger.ssm_offvtx_track1_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_score_p_bck", &tagger.ssm_offvtx_track1_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_score_p_bck", &tagger.ssm_offvtx_track1_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_score_e_bck", &tagger.ssm_offvtx_track1_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_score_e_bck", &tagger.ssm_offvtx_track1_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_length", &tagger.ssm_offvtx_track1_length);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_length", &tagger.ssm_offvtx_track1_length);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_direct_length", &tagger.ssm_offvtx_track1_direct_length);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_direct_length", &tagger.ssm_offvtx_track1_direct_length);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_max_dev", &tagger.ssm_offvtx_track1_max_dev);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_max_dev", &tagger.ssm_offvtx_track1_max_dev);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_kine_energy_range", &tagger.ssm_offvtx_track1_kine_energy_range);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_kine_energy_range", &tagger.ssm_offvtx_track1_kine_energy_range);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_kine_energy_range_mu", &tagger.ssm_offvtx_track1_kine_energy_range_mu);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_kine_energy_range_mu", &tagger.ssm_offvtx_track1_kine_energy_range_mu);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_kine_energy_range_p", &tagger.ssm_offvtx_track1_kine_energy_range_p);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_kine_energy_range_p", &tagger.ssm_offvtx_track1_kine_energy_range_p);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_kine_energy_range_e", &tagger.ssm_offvtx_track1_kine_energy_range_e);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_kine_energy_range_e", &tagger.ssm_offvtx_track1_kine_energy_range_e);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_kine_energy_cal", &tagger.ssm_offvtx_track1_kine_energy_cal);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_kine_energy_cal", &tagger.ssm_offvtx_track1_kine_energy_cal);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_medium_dq_dx", &tagger.ssm_offvtx_track1_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_medium_dq_dx", &tagger.ssm_offvtx_track1_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_x_dir", &tagger.ssm_offvtx_track1_x_dir);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_x_dir", &tagger.ssm_offvtx_track1_x_dir);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_y_dir", &tagger.ssm_offvtx_track1_y_dir);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_y_dir", &tagger.ssm_offvtx_track1_y_dir);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_z_dir", &tagger.ssm_offvtx_track1_z_dir);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_z_dir", &tagger.ssm_offvtx_track1_z_dir);
  reader_kdar_lowE.AddVariable("ssm_offvtx_track1_dist_mainvtx", &tagger.ssm_offvtx_track1_dist_mainvtx);
  reader_kdar_hiE.AddVariable("ssm_offvtx_track1_dist_mainvtx", &tagger.ssm_offvtx_track1_dist_mainvtx);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_pdg_offvtx", &tagger.ssm_offvtx_shw1_pdg_offvtx);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_pdg_offvtx", &tagger.ssm_offvtx_shw1_pdg_offvtx);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_score_mu_fwd", &tagger.ssm_offvtx_shw1_score_mu_fwd);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_score_mu_fwd", &tagger.ssm_offvtx_shw1_score_mu_fwd);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_score_p_fwd", &tagger.ssm_offvtx_shw1_score_p_fwd);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_score_p_fwd", &tagger.ssm_offvtx_shw1_score_p_fwd);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_score_e_fwd", &tagger.ssm_offvtx_shw1_score_e_fwd);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_score_e_fwd", &tagger.ssm_offvtx_shw1_score_e_fwd);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_score_mu_bck", &tagger.ssm_offvtx_shw1_score_mu_bck);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_score_mu_bck", &tagger.ssm_offvtx_shw1_score_mu_bck);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_score_p_bck", &tagger.ssm_offvtx_shw1_score_p_bck);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_score_p_bck", &tagger.ssm_offvtx_shw1_score_p_bck);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_score_e_bck", &tagger.ssm_offvtx_shw1_score_e_bck);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_score_e_bck", &tagger.ssm_offvtx_shw1_score_e_bck);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_length", &tagger.ssm_offvtx_shw1_length);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_length", &tagger.ssm_offvtx_shw1_length);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_direct_length", &tagger.ssm_offvtx_shw1_direct_length);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_direct_length", &tagger.ssm_offvtx_shw1_direct_length);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_max_dev", &tagger.ssm_offvtx_shw1_max_dev);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_max_dev", &tagger.ssm_offvtx_shw1_max_dev);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_kine_energy_best", &tagger.ssm_offvtx_shw1_kine_energy_best);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_kine_energy_best", &tagger.ssm_offvtx_shw1_kine_energy_best);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_kine_energy_range", &tagger.ssm_offvtx_shw1_kine_energy_range);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_kine_energy_range", &tagger.ssm_offvtx_shw1_kine_energy_range);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_kine_energy_range_mu", &tagger.ssm_offvtx_shw1_kine_energy_range_mu);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_kine_energy_range_mu", &tagger.ssm_offvtx_shw1_kine_energy_range_mu);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_kine_energy_range_p", &tagger.ssm_offvtx_shw1_kine_energy_range_p);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_kine_energy_range_p", &tagger.ssm_offvtx_shw1_kine_energy_range_p);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_kine_energy_range_e", &tagger.ssm_offvtx_shw1_kine_energy_range_e);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_kine_energy_range_e", &tagger.ssm_offvtx_shw1_kine_energy_range_e);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_kine_energy_cal", &tagger.ssm_offvtx_shw1_kine_energy_cal);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_kine_energy_cal", &tagger.ssm_offvtx_shw1_kine_energy_cal);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_medium_dq_dx", &tagger.ssm_offvtx_shw1_medium_dq_dx);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_medium_dq_dx", &tagger.ssm_offvtx_shw1_medium_dq_dx);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_x_dir", &tagger.ssm_offvtx_shw1_x_dir);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_x_dir", &tagger.ssm_offvtx_shw1_x_dir);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_y_dir", &tagger.ssm_offvtx_shw1_y_dir);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_y_dir", &tagger.ssm_offvtx_shw1_y_dir);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_z_dir", &tagger.ssm_offvtx_shw1_z_dir);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_z_dir", &tagger.ssm_offvtx_shw1_z_dir);
  reader_kdar_lowE.AddVariable("ssm_offvtx_shw1_dist_mainvtx", &tagger.ssm_offvtx_shw1_dist_mainvtx);
  reader_kdar_hiE.AddVariable("ssm_offvtx_shw1_dist_mainvtx", &tagger.ssm_offvtx_shw1_dist_mainvtx);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_mass", &tagger.ssm_kine_pio_mass);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_mass", &tagger.ssm_kine_pio_mass);
  float temp_ssm_kine_pio_flag;
  reader_kdar_lowE.AddVariable("ssm_kine_pio_flag", &temp_ssm_kine_pio_flag);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_flag", &temp_ssm_kine_pio_flag);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_vtx_dis", &tagger.ssm_kine_pio_vtx_dis);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_vtx_dis", &tagger.ssm_kine_pio_vtx_dis);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_energy_1", &tagger.ssm_kine_pio_energy_1);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_energy_1", &tagger.ssm_kine_pio_energy_1);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_theta_1", &tagger.ssm_kine_pio_theta_1);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_theta_1", &tagger.ssm_kine_pio_theta_1);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_phi_1", &tagger.ssm_kine_pio_phi_1);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_phi_1", &tagger.ssm_kine_pio_phi_1);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_dis_1", &tagger.ssm_kine_pio_dis_1);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_dis_1", &tagger.ssm_kine_pio_dis_1);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_energy_2", &tagger.ssm_kine_pio_energy_2);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_energy_2", &tagger.ssm_kine_pio_energy_2);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_theta_2", &tagger.ssm_kine_pio_theta_2);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_theta_2", &tagger.ssm_kine_pio_theta_2);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_phi_2", &tagger.ssm_kine_pio_phi_2);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_phi_2", &tagger.ssm_kine_pio_phi_2);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_dis_2", &tagger.ssm_kine_pio_dis_2);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_dis_2", &tagger.ssm_kine_pio_dis_2);
  reader_kdar_lowE.AddVariable("ssm_kine_pio_angle", &tagger.ssm_kine_pio_angle);
  reader_kdar_hiE.AddVariable("ssm_kine_pio_angle", &tagger.ssm_kine_pio_angle);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_1", &tagger.ssm_cosmict_flag_1);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_1", &tagger.ssm_cosmict_flag_1);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_2", &tagger.ssm_cosmict_flag_2);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_2", &tagger.ssm_cosmict_flag_2);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_3", &tagger.ssm_cosmict_flag_3);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_3", &tagger.ssm_cosmict_flag_3);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_4", &tagger.ssm_cosmict_flag_4);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_4", &tagger.ssm_cosmict_flag_4);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_5", &tagger.ssm_cosmict_flag_5);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_5", &tagger.ssm_cosmict_flag_5);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_6", &tagger.ssm_cosmict_flag_6);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_6", &tagger.ssm_cosmict_flag_6);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_7", &tagger.ssm_cosmict_flag_7);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_7", &tagger.ssm_cosmict_flag_7);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_8", &tagger.ssm_cosmict_flag_8);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_8", &tagger.ssm_cosmict_flag_8);
  reader_kdar_lowE.AddVariable("ssm_cosmict_flag_9", &tagger.ssm_cosmict_flag_9);
  reader_kdar_hiE.AddVariable("ssm_cosmict_flag_9", &tagger.ssm_cosmict_flag_9);
  reader_kdar_lowE.AddVariable("lm_cluster_length", &eval.lm_cluster_length);
  reader_kdar_hiE.AddVariable("lm_cluster_length", &eval.lm_cluster_length);
  
  reader_kdar_lowE.BookMVA( "MyBDT", "weights/kdar_lowE.xml");
  reader_kdar_hiE.BookMVA( "MyBDT", "weights/kdar_hiE.xml");


  TMVA::Reader reader_pi_veto;
  TMVA::Reader reader_mu_veto;
  TMVA::Reader reader_el_veto;
  TMVA::Reader reader_p_veto;
  TMVA::Reader reader_n_veto;
  float flag_has_prim_tracks=0;
  float reco_Emuon=0;
  TMVA::Reader reader_VtxAct_bdt;
  float temp_pi_veto_score = -999;
  float temp_mu_veto_score = -999;
  float temp_el_veto_score = -999;
  float temp_p_veto_score = -999;
  float temp_n_veto_score = -999;
  float temp_all_veto_score = -999;
  TMVA::Reader reader_all_veto;
  if(flag_spbdt){
    reader_pi_veto.AddVariable("spacepoints_q_0", &particle_info.spacepoints_q_0);
    reader_pi_veto.AddVariable("spacepoints_q_1", &particle_info.spacepoints_q_1);
    reader_pi_veto.AddVariable("spacepoints_q_2", &particle_info.spacepoints_q_2);
    reader_pi_veto.AddVariable("spacepoints_q_3", &particle_info.spacepoints_q_3);
    reader_pi_veto.AddVariable("spacepoints_q_4", &particle_info.spacepoints_q_4);
    reader_pi_veto.AddVariable("spacepoints_q_5", &particle_info.spacepoints_q_5);
    reader_pi_veto.AddVariable("spacepoints_q_6", &particle_info.spacepoints_q_6);
    reader_pi_veto.AddVariable("spacepoints_q_7", &particle_info.spacepoints_q_7);
    reader_pi_veto.AddVariable("spacepoints_q_8", &particle_info.spacepoints_q_8);
    reader_pi_veto.AddVariable("spacepoints_q_9", &particle_info.spacepoints_q_9);
    reader_pi_veto.AddVariable("spacepoints_q_10", &particle_info.spacepoints_q_10);
    reader_pi_veto.AddVariable("spacepoints_q_11", &particle_info.spacepoints_q_11);
    reader_pi_veto.AddVariable("spacepoints_q_12", &particle_info.spacepoints_q_12);
    reader_pi_veto.AddVariable("spacepoints_q_13", &particle_info.spacepoints_q_13);
    reader_pi_veto.AddVariable("spacepoints_q_14", &particle_info.spacepoints_q_14);
    reader_pi_veto.AddVariable("spacepoints_q_bck_0", &particle_info.spacepoints_q_bck_0);
    reader_pi_veto.AddVariable("spacepoints_q_bck_1", &particle_info.spacepoints_q_bck_1);
    reader_pi_veto.AddVariable("spacepoints_q_bck_2", &particle_info.spacepoints_q_bck_2);
    reader_pi_veto.AddVariable("spacepoints_q_bck_3", &particle_info.spacepoints_q_bck_3);
    reader_pi_veto.AddVariable("spacepoints_q_bck_4", &particle_info.spacepoints_q_bck_4);
    reader_pi_veto.AddVariable("spacepoints_q_bck_5", &particle_info.spacepoints_q_bck_5);
    reader_pi_veto.AddVariable("spacepoints_q_bck_6", &particle_info.spacepoints_q_bck_6);
    reader_pi_veto.AddVariable("spacepoints_q_bck_7", &particle_info.spacepoints_q_bck_7);
    reader_pi_veto.AddVariable("spacepoints_q_bck_8", &particle_info.spacepoints_q_bck_8);
    reader_pi_veto.AddVariable("spacepoints_q_bck_9", &particle_info.spacepoints_q_bck_9);
    reader_pi_veto.AddVariable("spacepoints_q_bck_10", &particle_info.spacepoints_q_bck_10);
    reader_pi_veto.AddVariable("spacepoints_q_bck_11", &particle_info.spacepoints_q_bck_11);
    reader_pi_veto.AddVariable("spacepoints_q_med", &particle_info.spacepoints_q_med);
    reader_pi_veto.AddVariable("flag_has_daught", &particle_info.flag_has_daught);
    reader_pi_veto.AddVariable("flag_has_daught_p", &particle_info.flag_has_daught_p);
    reader_pi_veto.AddVariable("flag_has_daught_el", &particle_info.flag_has_daught_el);
    reader_pi_veto.AddVariable("flag_has_daught_pi", &particle_info.flag_has_daught_pi);
    reader_pi_veto.AddVariable("reco_larpid_pidScore_el", &particle_info.reco_larpid_pidScore_el);
    reader_pi_veto.AddVariable("reco_larpid_pidScore_ph", &particle_info.reco_larpid_pidScore_ph);
    reader_pi_veto.AddVariable("reco_larpid_pidScore_mu", &particle_info.reco_larpid_pidScore_mu);
    reader_pi_veto.AddVariable("reco_larpid_pidScore_pr", &particle_info.reco_larpid_pidScore_pr);
    reader_pi_veto.AddVariable("reco_larpid_pidScore_pi", &particle_info.reco_larpid_pidScore_pi);
    reader_pi_veto.AddVariable("reco_larpid_proccess", &particle_info.reco_larpid_proccess);
    reader_pi_veto.AddVariable("flag_is_contained", &particle_info.flag_is_contained);
    reader_pi_veto.AddVariable("reco_pdg_list", &particle_info.reco_pdg);
    reader_pi_veto.AddVariable("track_len_ratio", &particle_info.track_len_ratio);
    reader_pi_veto.BookMVA( "MyBDT", "weights/pi_veto.xml");

    reader_mu_veto.AddVariable("spacepoints_q_0", &particle_info.spacepoints_q_0);
    reader_mu_veto.AddVariable("spacepoints_q_1", &particle_info.spacepoints_q_1);
    reader_mu_veto.AddVariable("spacepoints_q_2", &particle_info.spacepoints_q_2);
    reader_mu_veto.AddVariable("spacepoints_q_3", &particle_info.spacepoints_q_3);
    reader_mu_veto.AddVariable("spacepoints_q_4", &particle_info.spacepoints_q_4);
    reader_mu_veto.AddVariable("spacepoints_q_5", &particle_info.spacepoints_q_5);
    reader_mu_veto.AddVariable("spacepoints_q_6", &particle_info.spacepoints_q_6);
    reader_mu_veto.AddVariable("spacepoints_q_7", &particle_info.spacepoints_q_7);
    reader_mu_veto.AddVariable("spacepoints_q_8", &particle_info.spacepoints_q_8);
    reader_mu_veto.AddVariable("spacepoints_q_9", &particle_info.spacepoints_q_9);
    reader_mu_veto.AddVariable("spacepoints_q_10", &particle_info.spacepoints_q_10);
    reader_mu_veto.AddVariable("spacepoints_q_11", &particle_info.spacepoints_q_11);
    reader_mu_veto.AddVariable("spacepoints_q_12", &particle_info.spacepoints_q_12);
    reader_mu_veto.AddVariable("spacepoints_q_13", &particle_info.spacepoints_q_13);
    reader_mu_veto.AddVariable("spacepoints_q_14", &particle_info.spacepoints_q_14);
    reader_mu_veto.AddVariable("spacepoints_q_bck_0", &particle_info.spacepoints_q_bck_0);
    reader_mu_veto.AddVariable("spacepoints_q_bck_1", &particle_info.spacepoints_q_bck_1);
    reader_mu_veto.AddVariable("spacepoints_q_bck_2", &particle_info.spacepoints_q_bck_2);
    reader_mu_veto.AddVariable("spacepoints_q_bck_3", &particle_info.spacepoints_q_bck_3);
    reader_mu_veto.AddVariable("spacepoints_q_bck_4", &particle_info.spacepoints_q_bck_4);
    reader_mu_veto.AddVariable("spacepoints_q_bck_5", &particle_info.spacepoints_q_bck_5);
    reader_mu_veto.AddVariable("spacepoints_q_bck_6", &particle_info.spacepoints_q_bck_6);
    reader_mu_veto.AddVariable("spacepoints_q_bck_7", &particle_info.spacepoints_q_bck_7);
    reader_mu_veto.AddVariable("spacepoints_q_bck_8", &particle_info.spacepoints_q_bck_8);
    reader_mu_veto.AddVariable("spacepoints_q_bck_9", &particle_info.spacepoints_q_bck_9);
    reader_mu_veto.AddVariable("spacepoints_q_bck_10", &particle_info.spacepoints_q_bck_10);
    reader_mu_veto.AddVariable("spacepoints_q_bck_11", &particle_info.spacepoints_q_bck_11);
    reader_mu_veto.AddVariable("spacepoints_q_med", &particle_info.spacepoints_q_med);
    reader_mu_veto.AddVariable("flag_has_daught", &particle_info.flag_has_daught);
    reader_mu_veto.AddVariable("flag_has_daught_p", &particle_info.flag_has_daught_p);
    reader_mu_veto.AddVariable("flag_has_daught_el", &particle_info.flag_has_daught_el);
    reader_mu_veto.AddVariable("flag_has_daught_pi", &particle_info.flag_has_daught_pi);
    reader_mu_veto.AddVariable("reco_larpid_pidScore_el", &particle_info.reco_larpid_pidScore_el);
    reader_mu_veto.AddVariable("reco_larpid_pidScore_ph", &particle_info.reco_larpid_pidScore_ph);
    reader_mu_veto.AddVariable("reco_larpid_pidScore_mu", &particle_info.reco_larpid_pidScore_mu);
    reader_mu_veto.AddVariable("reco_larpid_pidScore_pr", &particle_info.reco_larpid_pidScore_pr);
    reader_mu_veto.AddVariable("reco_larpid_pidScore_pi", &particle_info.reco_larpid_pidScore_pi);
    reader_mu_veto.AddVariable("reco_larpid_proccess", &particle_info.reco_larpid_proccess);
    reader_mu_veto.AddVariable("flag_is_contained", &particle_info.flag_is_contained);
    reader_mu_veto.AddVariable("reco_pdg_list", &particle_info.reco_pdg);
    reader_mu_veto.AddVariable("track_len_ratio", &particle_info.track_len_ratio);
    reader_mu_veto.BookMVA( "MyBDT", "weights/mu_veto.xml");

    reader_el_veto.AddVariable("spacepoints_q_0", &particle_info.spacepoints_q_0);
    reader_el_veto.AddVariable("spacepoints_q_1", &particle_info.spacepoints_q_1);
    reader_el_veto.AddVariable("spacepoints_q_2", &particle_info.spacepoints_q_2);
    reader_el_veto.AddVariable("spacepoints_q_3", &particle_info.spacepoints_q_3);
    reader_el_veto.AddVariable("spacepoints_q_4", &particle_info.spacepoints_q_4);
    reader_el_veto.AddVariable("spacepoints_q_5", &particle_info.spacepoints_q_5);
    reader_el_veto.AddVariable("spacepoints_q_6", &particle_info.spacepoints_q_6);
    reader_el_veto.AddVariable("spacepoints_q_7", &particle_info.spacepoints_q_7);
    reader_el_veto.AddVariable("spacepoints_q_8", &particle_info.spacepoints_q_8);
    reader_el_veto.AddVariable("spacepoints_q_9", &particle_info.spacepoints_q_9);
    reader_el_veto.AddVariable("spacepoints_q_10", &particle_info.spacepoints_q_10);
    reader_el_veto.AddVariable("spacepoints_q_11", &particle_info.spacepoints_q_11);
    reader_el_veto.AddVariable("spacepoints_q_12", &particle_info.spacepoints_q_12);
    reader_el_veto.AddVariable("spacepoints_q_13", &particle_info.spacepoints_q_13);
    reader_el_veto.AddVariable("spacepoints_q_14", &particle_info.spacepoints_q_14);
    reader_el_veto.AddVariable("spacepoints_q_bck_0", &particle_info.spacepoints_q_bck_0);
    reader_el_veto.AddVariable("spacepoints_q_bck_1", &particle_info.spacepoints_q_bck_1);
    reader_el_veto.AddVariable("spacepoints_q_bck_2", &particle_info.spacepoints_q_bck_2);
    reader_el_veto.AddVariable("spacepoints_q_bck_3", &particle_info.spacepoints_q_bck_3);
    reader_el_veto.AddVariable("spacepoints_q_bck_4", &particle_info.spacepoints_q_bck_4);
    reader_el_veto.AddVariable("spacepoints_q_bck_5", &particle_info.spacepoints_q_bck_5);
    reader_el_veto.AddVariable("spacepoints_q_bck_6", &particle_info.spacepoints_q_bck_6);
    reader_el_veto.AddVariable("spacepoints_q_bck_7", &particle_info.spacepoints_q_bck_7);
    reader_el_veto.AddVariable("spacepoints_q_bck_8", &particle_info.spacepoints_q_bck_8);
    reader_el_veto.AddVariable("spacepoints_q_bck_9", &particle_info.spacepoints_q_bck_9);
    reader_el_veto.AddVariable("spacepoints_q_bck_10", &particle_info.spacepoints_q_bck_10);
    reader_el_veto.AddVariable("spacepoints_q_bck_11", &particle_info.spacepoints_q_bck_11);
    reader_el_veto.AddVariable("spacepoints_q_med", &particle_info.spacepoints_q_med);
    reader_el_veto.AddVariable("flag_has_daught", &particle_info.flag_has_daught);
    reader_el_veto.AddVariable("flag_has_daught_p", &particle_info.flag_has_daught_p);
    reader_el_veto.AddVariable("flag_has_daught_el", &particle_info.flag_has_daught_el);
    reader_el_veto.AddVariable("flag_has_daught_pi", &particle_info.flag_has_daught_pi);
    reader_el_veto.AddVariable("reco_larpid_pidScore_el", &particle_info.reco_larpid_pidScore_el);
    reader_el_veto.AddVariable("reco_larpid_pidScore_ph", &particle_info.reco_larpid_pidScore_ph);
    reader_el_veto.AddVariable("reco_larpid_pidScore_mu", &particle_info.reco_larpid_pidScore_mu);
    reader_el_veto.AddVariable("reco_larpid_pidScore_pr", &particle_info.reco_larpid_pidScore_pr);
    reader_el_veto.AddVariable("reco_larpid_pidScore_pi", &particle_info.reco_larpid_pidScore_pi);
    reader_el_veto.AddVariable("reco_larpid_proccess", &particle_info.reco_larpid_proccess);
    reader_el_veto.AddVariable("flag_is_contained", &particle_info.flag_is_contained);
    reader_el_veto.AddVariable("reco_pdg_list", &particle_info.reco_pdg);
    reader_el_veto.AddVariable("track_len_ratio", &particle_info.track_len_ratio);
    reader_el_veto.BookMVA( "MyBDT", "weights/el_veto.xml");

    reader_p_veto.AddVariable("spacepoints_q_0", &particle_info.spacepoints_q_0);
    reader_p_veto.AddVariable("spacepoints_q_1", &particle_info.spacepoints_q_1);
    reader_p_veto.AddVariable("spacepoints_q_2", &particle_info.spacepoints_q_2);
    reader_p_veto.AddVariable("spacepoints_q_3", &particle_info.spacepoints_q_3);
    reader_p_veto.AddVariable("spacepoints_q_4", &particle_info.spacepoints_q_4);
    reader_p_veto.AddVariable("spacepoints_q_5", &particle_info.spacepoints_q_5);
    reader_p_veto.AddVariable("spacepoints_q_6", &particle_info.spacepoints_q_6);
    reader_p_veto.AddVariable("spacepoints_q_7", &particle_info.spacepoints_q_7);
    reader_p_veto.AddVariable("spacepoints_q_8", &particle_info.spacepoints_q_8);
    reader_p_veto.AddVariable("spacepoints_q_9", &particle_info.spacepoints_q_9);
    reader_p_veto.AddVariable("spacepoints_q_10", &particle_info.spacepoints_q_10);
    reader_p_veto.AddVariable("spacepoints_q_11", &particle_info.spacepoints_q_11);
    reader_p_veto.AddVariable("spacepoints_q_12", &particle_info.spacepoints_q_12);
    reader_p_veto.AddVariable("spacepoints_q_13", &particle_info.spacepoints_q_13);
    reader_p_veto.AddVariable("spacepoints_q_14", &particle_info.spacepoints_q_14);
    reader_p_veto.AddVariable("spacepoints_q_bck_0", &particle_info.spacepoints_q_bck_0);
    reader_p_veto.AddVariable("spacepoints_q_bck_1", &particle_info.spacepoints_q_bck_1);
    reader_p_veto.AddVariable("spacepoints_q_bck_2", &particle_info.spacepoints_q_bck_2);
    reader_p_veto.AddVariable("spacepoints_q_bck_3", &particle_info.spacepoints_q_bck_3);
    reader_p_veto.AddVariable("spacepoints_q_bck_4", &particle_info.spacepoints_q_bck_4);
    reader_p_veto.AddVariable("spacepoints_q_bck_5", &particle_info.spacepoints_q_bck_5);
    reader_p_veto.AddVariable("spacepoints_q_bck_6", &particle_info.spacepoints_q_bck_6);
    reader_p_veto.AddVariable("spacepoints_q_bck_7", &particle_info.spacepoints_q_bck_7);
    reader_p_veto.AddVariable("spacepoints_q_bck_8", &particle_info.spacepoints_q_bck_8);
    reader_p_veto.AddVariable("spacepoints_q_bck_9", &particle_info.spacepoints_q_bck_9);
    reader_p_veto.AddVariable("spacepoints_q_bck_10", &particle_info.spacepoints_q_bck_10);
    reader_p_veto.AddVariable("spacepoints_q_bck_11", &particle_info.spacepoints_q_bck_11);
    reader_p_veto.AddVariable("spacepoints_q_med", &particle_info.spacepoints_q_med);
    reader_p_veto.AddVariable("flag_has_daught", &particle_info.flag_has_daught);
    reader_p_veto.AddVariable("flag_has_daught_p", &particle_info.flag_has_daught_p);
    reader_p_veto.AddVariable("flag_has_daught_el", &particle_info.flag_has_daught_el);
    reader_p_veto.AddVariable("flag_has_daught_pi", &particle_info.flag_has_daught_pi);
    reader_p_veto.AddVariable("reco_larpid_pidScore_el", &particle_info.reco_larpid_pidScore_el);
    reader_p_veto.AddVariable("reco_larpid_pidScore_ph", &particle_info.reco_larpid_pidScore_ph);
    reader_p_veto.AddVariable("reco_larpid_pidScore_mu", &particle_info.reco_larpid_pidScore_mu);
    reader_p_veto.AddVariable("reco_larpid_pidScore_pr", &particle_info.reco_larpid_pidScore_pr);
    reader_p_veto.AddVariable("reco_larpid_pidScore_pi", &particle_info.reco_larpid_pidScore_pi);
    reader_p_veto.AddVariable("reco_larpid_proccess", &particle_info.reco_larpid_proccess);
    reader_p_veto.AddVariable("flag_is_contained", &particle_info.flag_is_contained);
    reader_p_veto.AddVariable("reco_pdg_list", &particle_info.reco_pdg);
    reader_p_veto.AddVariable("track_len_ratio", &particle_info.track_len_ratio);
    reader_p_veto.BookMVA( "MyBDT", "weights/p_veto.xml");

    reader_n_veto.AddVariable("spacepoints_q_0", &particle_info.spacepoints_q_0);
    reader_n_veto.AddVariable("spacepoints_q_1", &particle_info.spacepoints_q_1);
    reader_n_veto.AddVariable("spacepoints_q_2", &particle_info.spacepoints_q_2);
    reader_n_veto.AddVariable("spacepoints_q_3", &particle_info.spacepoints_q_3);
    reader_n_veto.AddVariable("spacepoints_q_4", &particle_info.spacepoints_q_4);
    reader_n_veto.AddVariable("spacepoints_q_5", &particle_info.spacepoints_q_5);
    reader_n_veto.AddVariable("spacepoints_q_6", &particle_info.spacepoints_q_6);
    reader_n_veto.AddVariable("spacepoints_q_7", &particle_info.spacepoints_q_7);
    reader_n_veto.AddVariable("spacepoints_q_8", &particle_info.spacepoints_q_8);
    reader_n_veto.AddVariable("spacepoints_q_9", &particle_info.spacepoints_q_9);
    reader_n_veto.AddVariable("spacepoints_q_10", &particle_info.spacepoints_q_10);
    reader_n_veto.AddVariable("spacepoints_q_11", &particle_info.spacepoints_q_11);
    reader_n_veto.AddVariable("spacepoints_q_12", &particle_info.spacepoints_q_12);
    reader_n_veto.AddVariable("spacepoints_q_13", &particle_info.spacepoints_q_13);
    reader_n_veto.AddVariable("spacepoints_q_14", &particle_info.spacepoints_q_14);
    reader_n_veto.AddVariable("spacepoints_q_bck_0", &particle_info.spacepoints_q_bck_0);
    reader_n_veto.AddVariable("spacepoints_q_bck_1", &particle_info.spacepoints_q_bck_1);
    reader_n_veto.AddVariable("spacepoints_q_bck_2", &particle_info.spacepoints_q_bck_2);
    reader_n_veto.AddVariable("spacepoints_q_bck_3", &particle_info.spacepoints_q_bck_3);
    reader_n_veto.AddVariable("spacepoints_q_bck_4", &particle_info.spacepoints_q_bck_4);
    reader_n_veto.AddVariable("spacepoints_q_bck_5", &particle_info.spacepoints_q_bck_5);
    reader_n_veto.AddVariable("spacepoints_q_bck_6", &particle_info.spacepoints_q_bck_6);
    reader_n_veto.AddVariable("spacepoints_q_bck_7", &particle_info.spacepoints_q_bck_7);
    reader_n_veto.AddVariable("spacepoints_q_bck_8", &particle_info.spacepoints_q_bck_8);
    reader_n_veto.AddVariable("spacepoints_q_bck_9", &particle_info.spacepoints_q_bck_9);
    reader_n_veto.AddVariable("spacepoints_q_bck_10", &particle_info.spacepoints_q_bck_10);
    reader_n_veto.AddVariable("spacepoints_q_bck_11", &particle_info.spacepoints_q_bck_11);
    reader_n_veto.AddVariable("spacepoints_q_med", &particle_info.spacepoints_q_med);
    reader_n_veto.AddVariable("flag_has_daught", &particle_info.flag_has_daught);
    reader_n_veto.AddVariable("flag_has_daught_p", &particle_info.flag_has_daught_p);
    reader_n_veto.AddVariable("flag_has_daught_el", &particle_info.flag_has_daught_el);
    reader_n_veto.AddVariable("flag_has_daught_pi", &particle_info.flag_has_daught_pi);
    reader_n_veto.AddVariable("reco_larpid_pidScore_el", &particle_info.reco_larpid_pidScore_el);
    reader_n_veto.AddVariable("reco_larpid_pidScore_ph", &particle_info.reco_larpid_pidScore_ph);
    reader_n_veto.AddVariable("reco_larpid_pidScore_mu", &particle_info.reco_larpid_pidScore_mu);
    reader_n_veto.AddVariable("reco_larpid_pidScore_pr", &particle_info.reco_larpid_pidScore_pr);
    reader_n_veto.AddVariable("reco_larpid_pidScore_pi", &particle_info.reco_larpid_pidScore_pi);
    reader_n_veto.AddVariable("reco_larpid_proccess", &particle_info.reco_larpid_proccess);
    reader_n_veto.AddVariable("flag_is_contained", &particle_info.flag_is_contained);
    reader_n_veto.AddVariable("dist_to_vtx", &particle_info.dist_to_vtx);
    reader_n_veto.AddVariable("cos_theta", &particle_info.cos_theta);
    reader_n_veto.AddVariable("proximity", &particle_info.proximity);
    reader_n_veto.AddVariable("reco_pdg_list", &particle_info.reco_pdg);
    reader_n_veto.AddVariable("track_len_ratio", &particle_info.track_len_ratio);
    reader_n_veto.BookMVA( "MyBDT", "weights/n_veto.xml");

    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q0", &particle_info.spacepoints_q_0);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q1", &particle_info.spacepoints_q_1);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q2", &particle_info.spacepoints_q_2);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q3", &particle_info.spacepoints_q_3);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q4", &particle_info.spacepoints_q_4);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q5", &particle_info.spacepoints_q_5);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q6", &particle_info.spacepoints_q_6);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q7", &particle_info.spacepoints_q_7);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q8", &particle_info.spacepoints_q_8);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q9", &particle_info.spacepoints_q_9);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q10", &particle_info.spacepoints_q_10);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q11", &particle_info.spacepoints_q_11);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q12", &particle_info.spacepoints_q_12);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q0_bck", &particle_info.spacepoints_q_bck_0);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q1_bck", &particle_info.spacepoints_q_bck_1);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q2_bck", &particle_info.spacepoints_q_bck_2);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q3_bck", &particle_info.spacepoints_q_bck_3);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q4_bck", &particle_info.spacepoints_q_bck_4);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q5_bck", &particle_info.spacepoints_q_bck_5);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q6_bck", &particle_info.spacepoints_q_bck_6);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q7_bck", &particle_info.spacepoints_q_bck_7);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q8_bck", &particle_info.spacepoints_q_bck_8);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q9_bck", &particle_info.spacepoints_q_bck_9);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q10_bck", &particle_info.spacepoints_q_bck_10);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q11_bck", &particle_info.spacepoints_q_bck_11);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q12_bck", &particle_info.spacepoints_q_bck_12);
    reader_VtxAct_bdt.AddVariable("muon_spacepoints_q_med", &particle_info.spacepoints_q_med);
    reader_VtxAct_bdt.AddVariable("reco_Emuon", &reco_Emuon);
    reader_VtxAct_bdt.AddVariable("flag_has_prim_tracks", &flag_has_prim_tracks);
    reader_VtxAct_bdt.BookMVA( "MyBDT", "weights/VtxAct_BDT.xml");
  
    reader_all_veto.AddVariable("score_pi_veto", &temp_pi_veto_score);
    reader_all_veto.AddVariable("score_mu2_veto", &temp_mu_veto_score);
    reader_all_veto.AddVariable("score_el_veto", &temp_el_veto_score);
    reader_all_veto.AddVariable("score_p_veto", &temp_p_veto_score);
    reader_all_veto.AddVariable("reco_pdg_list", &particle_info.reco_pdg);
    reader_all_veto.AddVariable("flag_is_contained",&particle_info.flag_is_contained);
    reader_all_veto.BookMVA( "MyBDT", "weights/all_veto.xml");
  }

  std::map<std::pair<int, int>, int> map_rs_n;
  std::map<std::pair<int, int>, std::set<int> > map_rs_f1p5; // Reco 1.5
  std::map<std::pair<int, int>, std::set<int> > map_rs_f2stm; // Reco2 stm
  std::map<std::pair<int, int>, std::set<int> > map_rs_f2pr; // Reco2 Pattern recognition

  T_eval->SetBranchStatus("*",0);
  T_eval->SetBranchStatus("stm_eventtype",1);
  T_eval->SetBranchStatus("stm_lowenergy",1);
  T_eval->SetBranchStatus("stm_LM",1);
  T_eval->SetBranchStatus("stm_TGM",1);
  T_eval->SetBranchStatus("stm_STM",1);
  T_eval->SetBranchStatus("stm_FullDead",1);
  T_eval->SetBranchStatus("stm_clusterlength",1);
  T_eval->SetBranchStatus("match_found",1);
  T_eval->SetBranchStatus("run",1);
  T_eval->SetBranchStatus("subrun",1);
  T_eval->SetBranchStatus("event",1);
  if (T_eval->GetBranch("file_type")) T_eval->SetBranchStatus("file_type",1);

  if (T_eval->GetBranch("match_found_asInt")){
    T_eval->SetBranchStatus("match_found_asInt",1);
  }

  T_BDTvars->SetBranchStatus("*",0);
  T_BDTvars->SetBranchStatus("numu_cc_flag",1);

  int haveReco;
  if(T_lantern && remove_lantern_fails==1) T_lantern->SetBranchAddress("haveReco",&haveReco);

  std::set<std::pair<int,int> > remove_set;

  bool flag_presel = false;
  for (Int_t i=0;i!=T_eval->GetEntries();i++){

    T_eval->GetEntry(i);
    T_BDTvars->GetEntry(i);

    // Check if the Lantern container failed on this event, if so throw out the subrun.
    if(T_lantern) T_lantern->GetEntry(i);
    if(remove_lantern_fails==1 && haveReco==0){
      remove_set.insert(std::make_pair(eval.run, eval.subrun));
      continue;
    }

    // Remove runs if they are in the extra list provided 
    auto rs_it = remove_individual_run.find(eval.run);
    if (rs_it != remove_individual_run.end()) {
      // Removing all subruns in this run
      if ((*rs_it).second.at(0)==-1){
        remove_set.insert(std::make_pair(eval.run, eval.subrun));
        continue;
      }
      // Check if this subrun is in the list of ones to remove in the given run 
      auto s_it = std::find((*rs_it).second.begin(), (*rs_it).second.end(), eval.subrun);
      if (s_it != (*rs_it).second.end()) {
        remove_set.insert(std::make_pair(eval.run, eval.subrun));
        continue; 
      }
    }
 
    // Remove (or keep) BDT training runs.
    if (flag_check_run_subrun){
      if (flag_use_global_file_type){
	(*eval.file_type) = global_file_type;
      }
      auto it1 = map_type_run_subrun.find(*eval.file_type);

      if (it1 != map_type_run_subrun.end()){
	// removing run-subruns used to train the BDTs
	if ( it1->second.find(std::make_pair(eval.run, eval.subrun)) != it1->second.end() && flag_keep_only_bdt_train==0 ) {
	  remove_set.insert(std::make_pair(eval.run, eval.subrun));
	  continue;
	}
        // removing run-subruns NOT used to train the BDTs
        if ( it1->second.find(std::make_pair(eval.run, eval.subrun)) == it1->second.end() && flag_keep_only_bdt_train==1 ) {
          remove_set.insert(std::make_pair(eval.run, eval.subrun));
          continue;
        }
      }
      // removing run-subruns NOT used to train the BDTs
      else if(it1 == map_type_run_subrun.end() && flag_keep_only_bdt_train==1){
        remove_set.insert(std::make_pair(eval.run, eval.subrun));
        continue;
      }

      //      std::cout << flag_use_global_file_type << " " << *eval.file_type  << " " << eval.run << " " << eval.subrun << " " << remove_set.size() << std::endl;
    }

    /*
    // failed jobs ...
    int tmp_match_found = eval.match_found;
    if (eval.is_match_found_int) tmp_match_found = eval.match_found_asInt;

    flag_presel = false;
    if (tmp_match_found != 0 && eval.stm_eventtype != 0 && eval.stm_lowenergy ==0 && eval.stm_LM ==0 && eval.stm_TGM ==0 && eval.stm_STM==0 && eval.stm_FullDead == 0 && eval.stm_clusterlength >0) {
      flag_presel = true; // preselection ...
    }

    map_rs_n[std::make_pair(eval.run, eval.subrun)] ++;
    if (tmp_match_found == -1) map_rs_f1p5[std::make_pair(eval.run, eval.subrun)].insert(eval.event);
    if (tmp_match_found == 1 && eval.stm_lowenergy == -1) map_rs_f2stm[std::make_pair(eval.run, eval.subrun)].insert(eval.event);
    if (flag_presel && tagger.numu_cc_flag == -1) map_rs_f2pr[std::make_pair(eval.run, eval.subrun)].insert(eval.event);
    */
  }


  /*
  for (auto it = map_rs_f1p5.begin(); it!= map_rs_f1p5.end(); it++){
    if ( map_rs_f1p5[it->first].size()  > map_rs_n[it->first] * fail_percentage && map_rs_f1p5[it->first].size() != 1
	 || map_rs_f1p5[it->first].size()  > map_rs_n[it->first] * 0.33 && map_rs_f1p5[it->first].size() == 1)
      remove_set.insert(it->first);
  }
  for (auto it = map_rs_f2stm.begin(); it != map_rs_f2stm.end(); it++){
    if (map_rs_f2stm[it->first].size() > map_rs_n[it->first]* fail_percentage && map_rs_f2stm[it->first].size() != 1
	|| map_rs_f2stm[it->first].size() > map_rs_n[it->first]* 0.33 && map_rs_f2stm[it->first].size() == 1)
      remove_set.insert(it->first);
  }
  for (auto it = map_rs_f2pr.begin(); it != map_rs_f2pr.end(); it++){
    if (map_rs_f2pr[it->first].size() > map_rs_n[it->first] * fail_percentage && map_rs_f2pr[it->first].size() != 1 || map_rs_f2pr[it->first].size() > map_rs_n[it->first] * 0.33 && map_rs_f2pr[it->first].size() == 1
	)
      remove_set.insert(it->first);
  }
  */

  //  std::cout << remove_set.size() << std::endl;


  T_eval->SetBranchStatus("*",1);
  T_BDTvars->SetBranchStatus("*",1);
  T_spacepoints->SetBranchStatus("*",1);

  if(flag_set_samdef){
    t1->Branch("samdef", "TString", &samdef);
  }

  int nentries = T_BDTvars->GetEntries();
  std::cout<<"Begin looping over "<<nentries<<" events"<<std::endl;
  for (int i=0;i!=nentries;i++){

    if (i%10000 == 0) std::cout << i/1000 << " k " << std::setprecision(3) << double(i)/nentries*100. << " %"<< std::endl;

    eval.weight_change = false;
    T_BDTvars->GetEntry(i); temp_ssm_kine_pio_flag = tagger.ssm_kine_pio_flag;
    T_eval->GetEntry(i); tagger.match_isFC = eval.match_isFC;
    T_KINEvars->GetEntry(i); tagger.kine_reco_Enu = kine.kine_reco_Enu; temp_kine_pio_flag = kine.kine_pio_flag;
    T_PFeval->GetEntry(i);
    T_spacepoints->GetEntry(i);

    if (remove_set.find(std::make_pair(eval.run, eval.subrun)) != remove_set.end()) continue;

    tagger.br3_3_score     = cal_br3_3_bdt(0.3, tagger,  reader_br3_3, br3_3_v_energy,  br3_3_v_angle,  br3_3_v_dir_length, br3_3_v_length);
    tagger.br3_5_score     = cal_br3_5_bdt(0.42, tagger,  reader_br3_5, br3_5_v_dir_length, br3_5_v_total_length, br3_5_v_flag_avoid_muon_check, br3_5_v_n_seg, br3_5_v_angle, br3_5_v_sg_length, br3_5_v_energy, br3_5_v_n_main_segs, br3_5_v_n_segs, br3_5_v_shower_main_length, br3_5_v_shower_total_length);
    tagger.br3_6_score     = cal_br3_6_bdt(0.75, tagger, reader_br3_6, br3_6_v_angle, br3_6_v_angle1, br3_6_v_flag_shower_trajectory, br3_6_v_direct_length, br3_6_v_length, br3_6_v_n_other_vtx_segs, br3_6_v_energy);
    tagger.pio_2_score     = cal_pio_2_bdt(0.2,  tagger,  reader_pio_2, pio_2_v_dis2, pio_2_v_angle2, pio_2_v_acc_length);
    tagger.stw_2_score     = cal_stw_2_bdt(0.7, tagger, reader_stw_2, stw_2_v_medium_dQ_dx, stw_2_v_energy, stw_2_v_angle, stw_2_v_dir_length, stw_2_v_max_dQ_dx);
    tagger.stw_3_score     = cal_stw_3_bdt(0.5, tagger, reader_stw_3, stw_3_v_angle, stw_3_v_dir_length, stw_3_v_energy, stw_3_v_medium_dQ_dx);
    tagger.stw_4_score     = cal_stw_4_bdt(0.7, tagger, reader_stw_4, stw_4_v_angle, stw_4_v_dis, stw_4_v_energy);
    tagger.sig_1_score     = cal_sig_1_bdt(0.59, tagger,  reader_sig_1, sig_1_v_angle, sig_1_v_flag_single_shower, sig_1_v_energy, sig_1_v_energy_1);
    tagger.sig_2_score     = cal_sig_2_bdt(0.55, tagger, reader_sig_2, sig_2_v_energy, sig_2_v_shower_angle, sig_2_v_flag_single_shower, sig_2_v_medium_dQ_dx, sig_2_v_start_dQ_dx);
    tagger.lol_1_score     = cal_lol_1_bdt(0.85, tagger, reader_lol_1, lol_1_v_energy, lol_1_v_vtx_n_segs, lol_1_v_nseg, lol_1_v_angle);
    tagger.lol_2_score     = cal_lol_2_bdt(0.7, tagger, reader_lol_2,  lol_2_v_length, lol_2_v_angle, lol_2_v_type, lol_2_v_vtx_n_segs, lol_2_v_energy, lol_2_v_shower_main_length, lol_2_v_flag_dir_weak);
    tagger.tro_1_score     = cal_tro_1_bdt(0.28, tagger, reader_tro_1, tro_1_v_particle_type, tro_1_v_flag_dir_weak, tro_1_v_min_dis, tro_1_v_sg1_length,
					   tro_1_v_shower_main_length, tro_1_v_max_n_vtx_segs, tro_1_v_tmp_length, tro_1_v_medium_dQ_dx, tro_1_v_dQ_dx_cut, tro_1_v_flag_shower_topology);
    tagger.tro_2_score     = cal_tro_2_bdt(0.35, tagger, reader_tro_2, tro_2_v_energy, tro_2_v_stem_length, tro_2_v_iso_angle, tro_2_v_max_length, tro_2_v_angle);
    tagger.tro_4_score     = cal_tro_4_bdt(0.33, tagger, reader_tro_4, tro_4_v_dir2_mag, tro_4_v_angle, tro_4_v_angle1, tro_4_v_angle2, tro_4_v_length, tro_4_v_length1, tro_4_v_medium_dQ_dx, tro_4_v_end_dQ_dx, tro_4_v_energy, tro_4_v_shower_main_length, tro_4_v_flag_shower_trajectory);
    tagger.tro_5_score     = cal_tro_5_bdt(0.5, tagger, reader_tro_5, tro_5_v_max_angle, tro_5_v_min_angle, tro_5_v_max_length, tro_5_v_iso_angle, tro_5_v_n_vtx_segs, tro_5_v_min_count, tro_5_v_max_count, tro_5_v_energy);
    tagger.nue_score       = cal_bdts_xgboost( tagger,  reader);

    // BDT calculations
    tagger.numu_1_score = cal_numu_1_bdt(-0.4, tagger, reader_numu_1, numu_cc_flag_1,
					 numu_cc_1_particle_type,
					 numu_cc_1_length,
					 numu_cc_1_medium_dQ_dx,
					 numu_cc_1_dQ_dx_cut,
					 numu_cc_1_direct_length,
					 numu_cc_1_n_daughter_tracks,
					 numu_cc_1_n_daughter_all);
    tagger.numu_2_score = cal_numu_2_bdt(-0.1,tagger,reader_numu_2,
					 numu_cc_2_length,
					 numu_cc_2_total_length,
					 numu_cc_2_n_daughter_tracks,
					 numu_cc_2_n_daughter_all);
    tagger.cosmict_10_score = cal_cosmict_10_bdt(0.7, tagger, reader_cosmict_10,
						 cosmict_10_vtx_z,
						 cosmict_10_flag_shower,
						 cosmict_10_flag_dir_weak,
						 cosmict_10_angle_beam,
						 cosmict_10_length);
    if (std::isnan(tagger.cosmict_4_angle_beam)) tagger.cosmict_4_angle_beam = 0;
    if (std::isnan(tagger.cosmict_7_angle_beam)) tagger.cosmict_7_angle_beam = 0;
    if (std::isnan(tagger.cosmict_7_theta)) tagger.cosmict_7_theta = 0;
    if (std::isnan(tagger.cosmict_7_phi)) tagger.cosmict_7_phi = 0;

    tagger.numu_score = cal_numu_bdts_xgboost(tagger,reader_numu);

    // NC gamma ... add a NC BDT ... Lee
    tagger.nc_delta_score = cal_nc_delta_bdts_xgboost(tagger, reader_nc_delta);
    tagger.nc_delta_0track_score = cal_nc_delta_0track_bdts_xgboost(tagger, reader_nc_delta_0track);
    tagger.nc_delta_ntrack_score = cal_nc_delta_ntrack_bdts_xgboost(tagger, reader_nc_delta_ntrack);

    //    std::cout << tagger.nc_delta_0track_score << " " << tagger.nc_delta_ntrack_score << std::endl;

    tagger.nc_pio_score = cal_nc_pio_bdts_xgboost(tagger, reader_nc_pi0);
    // if (eval.run == 6467 && eval.event == 97
    // 	|| eval.run == 7026 && eval.event == 22180
    // 	|| eval.run == 7010 && eval.event == 25278
    // 	|| eval.run == 5531 && eval.event == 3160
    // 	|| eval.run == 7020 && eval.event == 32277
    // 	){
    // if (tagger.numu_cc_flag>=0)
    //   std::cout << i << " " << eval.run << " " << eval.event << " " << nc_pio_score << std::endl;
    // }
    //

    //hack
    //    tagger.nc_delta_score = -15;

    // Single Photon BDTs ... Erin
    float em_charge_scale = 1.0;
    //if(flag_data) em_charge_scale = 0.95;
    tagger.shw_sp_vec_median_dedx *= em_charge_scale;
    tagger.shw_sp_vec_mean_dedx   *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_0     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_1     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_2     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_3     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_4     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_5     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_6     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_7     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_8     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_9     *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_10    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_11    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_12    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_13    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_14    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_15    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_16    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_17    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_18    *= em_charge_scale;
    tagger.shw_sp_vec_dQ_dx_19    *= em_charge_scale;

    //uncomment to match old files
    temp_kine_pio_flag = 0;

    tagger.single_photon_numu_score = cal_single_photon_numu_bdts_xgboost(tagger, reader_single_photon_numu);
    tagger.single_photon_other_score = cal_single_photon_other_bdts_xgboost(tagger, reader_single_photon_other);
    tagger.single_photon_ncpi0_score = cal_single_photon_ncpi0_bdts_xgboost(tagger, reader_single_photon_ncpi0);
    tagger.single_photon_nue_score = cal_single_photon_nue_bdts_xgboost(tagger, reader_single_photon_nue);

    if (std::isnan(tagger.ssm_nu_angle_z)) tagger.ssm_nu_angle_z = 0;
    if (std::isnan(tagger.ssm_nu_angle_target)) tagger.ssm_nu_angle_target = 0;
    if (std::isnan(tagger.ssm_nu_angle_absorber)) tagger.ssm_nu_angle_absorber = 0;
    if (std::isnan(tagger.ssm_nu_angle_vertical)) tagger.ssm_nu_angle_vertical = 0;
    if (std::isnan(tagger.ssm_prim_nu_angle_z)) tagger.ssm_prim_nu_angle_z = 0;
    if (std::isnan(tagger.ssm_prim_nu_angle_target)) tagger.ssm_prim_nu_angle_target = 0;
    if (std::isnan(tagger.ssm_prim_nu_angle_absorber)) tagger.ssm_prim_nu_angle_absorber = 0;
    if (std::isnan(tagger.ssm_prim_nu_angle_vertical)) tagger.ssm_prim_nu_angle_vertical = 0;
    if (std::isnan(tagger.ssm_con_nu_angle_z)) tagger.ssm_con_nu_angle_z = 0;
    if (std::isnan(tagger.ssm_con_nu_angle_target)) tagger.ssm_con_nu_angle_target = 0;
    if (std::isnan(tagger.ssm_con_nu_angle_absorber)) tagger.ssm_con_nu_angle_absorber = 0;
    if (std::isnan(tagger.ssm_con_nu_angle_vertical)) tagger.ssm_con_nu_angle_vertical = 0;
    if (std::isnan(tagger.ssm_track_angle_z)) tagger.ssm_track_angle_z = 0;
    if (std::isnan(tagger.ssm_track_angle_target)) tagger.ssm_track_angle_target = 0;
    if (std::isnan(tagger.ssm_track_angle_absorber)) tagger.ssm_track_angle_absorber = 0;
    if (std::isnan(tagger.ssm_track_angle_vertical)) tagger.ssm_track_angle_vertical = 0;
    if (std::isnan(tagger.ssm_kine_pio_mass)) tagger.ssm_kine_pio_mass = 0;
    if (std::isnan(tagger.ssm_kine_pio_vtx_dis)) tagger.ssm_kine_pio_vtx_dis = 0;
    if (std::isnan(tagger.ssm_kine_pio_theta_1)) tagger.ssm_kine_pio_theta_1 = 0;
    if (std::isnan(tagger.ssm_kine_pio_theta_2)) tagger.ssm_kine_pio_theta_2 = 0;
    if (std::isnan(tagger.ssm_kine_pio_phi_1)) tagger.ssm_kine_pio_phi_1 = 0;
    if (std::isnan(tagger.ssm_kine_pio_phi_2)) tagger.ssm_kine_pio_phi_2 = 0;
    if (std::isnan(tagger.ssm_kine_pio_dis_1)) tagger.ssm_kine_pio_dis_1 = 0;
    if (std::isnan(tagger.ssm_kine_pio_dis_2)) tagger.ssm_kine_pio_dis_2 = 0;
    if (std::isnan(tagger.ssm_kine_pio_angle)) tagger.ssm_kine_pio_angle = 0;
    tagger.ssm_kdar_score_lowE = cal_kdar_lowE_bdt_xgboost(tagger, eval, reader_kdar_lowE);
    tagger.ssm_kdar_score_hiE = cal_kdar_hiE_bdt_xgboost(tagger, eval, reader_kdar_hiE);

    tagger.pi_veto_score=-999;
    tagger.pi_veto_prim_score=-999;
    tagger.pi_veto_all_score=-999;
    tagger.mu_veto_score=-999;
    tagger.mu_veto_prim_score=-999;
    tagger.mu_veto_all_score=-999;
    tagger.el_veto_score=-999;
    tagger.el_veto_prim_score=-999;
    tagger.el_veto_all_score=-999;
    tagger.n_veto_score=-999;
    tagger.n_veto_nonprim_score=-999;
    tagger.n_veto_all_score=-999;
    tagger.all_veto_score=-999;
    tagger.VtxAct_bdt_score=-999;
    int prim_mu_index = -1;
    flag_has_prim_tracks=0;
    reco_Emuon=0;
    for(int part=0; part<pfeval.reco_Ntrack; part++){
      if(!flag_spbdt) break;
      temp_pi_veto_score = -999;
      temp_mu_veto_score = -999;
      temp_el_veto_score = -999;
      temp_p_veto_score = -999;
      temp_n_veto_score = -999;
      temp_all_veto_score = -999;
      create_particle(space_info, pfeval, particle_info, part, flag_data);
      if(particle_info.reco_pdg<0) continue;
      temp_pi_veto_score = cal_spacepoint_pi_veto(particle_info,reader_pi_veto);
      if(temp_pi_veto_score>tagger.pi_veto_all_score) tagger.pi_veto_all_score = temp_pi_veto_score;
      if(temp_pi_veto_score>tagger.pi_veto_prim_score && pfeval.reco_mother[part]==0) tagger.pi_veto_prim_score = temp_pi_veto_score;
      if(temp_pi_veto_score>tagger.pi_veto_score && pfeval.reco_mother[part]==0 && pfeval.reco_pdg[part]==211) tagger.pi_veto_score = temp_pi_veto_score;
//std::cout<<temp_pi_veto_score<<" "<<particle_info.reco_pdg<<" "<<particle_info.reco_momentum_0<<std::endl;
      temp_mu_veto_score = cal_spacepoint_mu_veto(particle_info,reader_mu_veto);
      if(temp_mu_veto_score>tagger.mu_veto_all_score) tagger.mu_veto_all_score = temp_mu_veto_score;
      if(temp_mu_veto_score>tagger.mu_veto_prim_score && pfeval.reco_mother[part]==0 && particle_info.flag_prim_mu==0) tagger.mu_veto_prim_score = temp_mu_veto_score;
      if(temp_mu_veto_score>tagger.mu_veto_score && pfeval.reco_mother[part]==0 && pfeval.reco_pdg[part]==13 && particle_info.flag_prim_mu==0) tagger.mu_veto_score = temp_mu_veto_score;

      temp_el_veto_score = cal_spacepoint_el_veto(particle_info,reader_el_veto);
      if(temp_el_veto_score>tagger.el_veto_all_score) tagger.el_veto_all_score = temp_el_veto_score;
      if(temp_el_veto_score>tagger.el_veto_prim_score && pfeval.reco_mother[part]==0) tagger.el_veto_prim_score = temp_el_veto_score;
      if(temp_el_veto_score>tagger.el_veto_score && pfeval.reco_mother[part]==0 && pfeval.reco_pdg[part]==11) tagger.el_veto_score = temp_el_veto_score;

      temp_p_veto_score = cal_spacepoint_p_veto(particle_info,reader_p_veto);
      if(temp_p_veto_score>tagger.p_veto_all_score) tagger.p_veto_all_score = temp_p_veto_score;
      if(temp_p_veto_score>tagger.p_veto_prim_score && pfeval.reco_mother[part]==0) tagger.p_veto_prim_score = temp_p_veto_score;
      if(temp_p_veto_score>tagger.p_veto_score && pfeval.reco_mother[part]==0 && pfeval.reco_pdg[part]==2212) tagger.p_veto_score = temp_p_veto_score;
//std::cout<<temp_p_veto_score<<" "<<particle_info.reco_pdg<<" "<<particle_info.reco_momentum_0<<std::endl;

      temp_n_veto_score = cal_spacepoint_n_veto(particle_info,reader_n_veto);
      if(temp_n_veto_score>tagger.n_veto_all_score) tagger.n_veto_all_score = temp_n_veto_score;
      if(temp_n_veto_score>tagger.n_veto_nonprim_score && pfeval.reco_mother[part]!=0) tagger.n_veto_nonprim_score = temp_n_veto_score;
      if(temp_n_veto_score>tagger.n_veto_score && pfeval.reco_mother[part]!=0 && (particle_info.reco_is_g_induced==1 || particle_info.reco_is_n_induced==1)) tagger.n_veto_score = temp_n_veto_score; 
  
      if(particle_info.flag_prim_mu==1){ prim_mu_index = part;}
      else{
        temp_pi_veto_score +=-0.03167;
        temp_mu_veto_score +=-0.12784585;
        temp_el_veto_score +=-0.5630117;
        temp_p_veto_score +=1.2690324;
        temp_n_veto_score +=-1.4071424;
        temp_all_veto_score = cal_spacepoint_all_veto(particle_info,reader_all_veto);
        if(temp_all_veto_score>tagger.all_veto_score) tagger.all_veto_score = temp_all_veto_score;
      }

      if(pfeval.reco_mother[part]==0 && (pfeval.reco_pdg[part]==2212 || pfeval.reco_pdg[part]==211) ) flag_has_prim_tracks=1;
    }
    if(prim_mu_index>=0){
      create_particle(space_info, pfeval, particle_info, prim_mu_index, flag_data);
      reco_Emuon = (particle_info.reco_momentum_3+0.1057)*1000;
      tagger.VtxAct_bdt_score = cal_VtxAct_bdt_score(particle_info,reader_VtxAct_bdt);
    }


    // limit the cut val ...
    if (std::isnan(eval.weight_spline) || std::isinf(eval.weight_spline) ||
	std::isnan(eval.weight_cv) || std::isinf(eval.weight_cv) ||
	eval.weight_spline * eval.weight_cv <=0 ||
	eval.weight_spline * eval.weight_cv > weight_cut_val){
      eval.weight_spline = 1;
      eval.weight_cv = 1;
      eval.weight_change = true;
    }

    if(flag_gibuu){
      eval.weight_spline = 1;
      eval.weight_cv = eval.truth_nuTime;
      eval.truth_nuTime = pfeval.truth_startXYZT[0][3]/1000;
      pfeval.truth_nuTime = pfeval.truth_startXYZT[0][3]/1000;
      pfeval.truth_nu_pos[3] = pfeval.truth_startXYZT[0][3];
      pfeval.mc_nu_pos[3] = pfeval.truth_startXYZT[0][3];
    }

    if (flag_data && skip_cut == 0){
      if (good_runlist_set.find(eval.run) == good_runlist_set.end()) continue;
      if (low_lifetime_set.find(eval.run) != low_lifetime_set.end()) continue;
      if (flag_numi && low_neutrino_count_numi_run2RHC_set.find(eval.run) != low_neutrino_count_numi_run2RHC_set.end()) continue;
      // bad run in run 1 due to beam filter bnb
      // if (eval.run <= 5367 && eval.run >= 5320) continue;
      // ext bnb in run 1, high rate
      if (eval.run>=7004 && eval.run <=7070) continue;
      // ext bnb in run 2, high rate not in good list anyway
      //      if ((eval.run>=10287 && eval.run <= 10304) || (eval.run>=12277 && eval.run <=12350)) continue;
      // ext bnb in run 2, low rate not in good list anyway
      // if ((eval.run>=9768 && eval.run <= 10070) || (eval.run>=10102 && eval.run <=10246)) continue;
      // bnb run 2 high rate
      if (eval.run >= 8321 && eval.run <=8404) continue;
      // bnb run 3 high rate
      if (eval.run >=15369 && eval.run <= 15402) continue;
    }
 
    t4->Fill();
    t1->Fill();
    t3->Fill();
    t5->Fill();

    //T_spacepoints->GetEntry(i);
    new_T_spacepoints->Fill();

    for(auto tree_it=wrangler.old_trees->begin(); tree_it!=wrangler.old_trees->end(); tree_it++){
        (*tree_it)->GetEntry(i);
    }
    for(auto tree_it=wrangler.new_trees->begin(); tree_it!=wrangler.new_trees->end(); tree_it++){
        (*tree_it)->Fill();
    }
    for(auto tree_it=wrangler_ex.old_trees->begin(); tree_it!=wrangler_ex.old_trees->end(); tree_it++){
        (*tree_it)->GetEntry(i);
    }
    for(auto tree_it=wrangler_ex.new_trees->begin(); tree_it!=wrangler_ex.new_trees->end(); tree_it++){
        (*tree_it)->Fill();
    }
    //    std::cout << pfeval.reco_daughters->size() << std::endl;
    //    break;
  }


  // Loop over each POT tree seperatly
  // Start with WireCell
  nentries = T_pot->GetEntries();
  std::cout<<"Begin looping over WC pot tree with "<<nentries<<" entries"<<std::endl;
  for (Int_t i=0;i!=nentries;i++){

    if (i%10000 == 0) std::cout << i/1000 << " k " << std::setprecision(3) << double(i)/nentries*100. << " %"<< std::endl;

    T_pot->GetEntry(i);

    if (remove_set.find(std::make_pair(pot.runNo, pot.subRunNo)) != remove_set.end()) continue;

    if (flag_data && skip_cut == 0){
      if (good_runlist_set.find(pot.runNo) == good_runlist_set.end()) continue;
      if (low_lifetime_set.find(pot.runNo) != low_lifetime_set.end()) continue;
      if (flag_numi && low_neutrino_count_numi_run2RHC_set.find(pot.runNo) != low_neutrino_count_numi_run2RHC_set.end()) continue;

      //bad run in run 1 due to beam filter bnb
      //if (pot.runNo <= 5367 && pot.runNo >= 5320) continue;
      // ext bnb in run 1, high rate
      if (pot.runNo >=7004 && pot.runNo <=7070) continue;
      // ext bnb in run 2, high rate not in good list anyway
      //if ((pot.runNo>=10287 && pot.runNo <= 10304) || (pot.runNo>=12277 && pot.runNo <=12350)) continue;
      // ext bnb in run 2, low rate not in good list anyway
      //if ((pot.runNo>=9768 && pot.runNo <= 10070) || (pot.runNo>=10102 && pot.runNo <=10246)) continue;
      // bnb run 2 high rate
      if (pot.runNo >= 8321 && pot.runNo <=8404) continue;
      // bnb run 3 high rate
      if (pot.runNo >=15369 && pot.runNo <= 15402) continue;
    }
    t2->Fill();
  }

  // Now the other trees
  for(auto pot_tree_it=wrangler_pot.pot_arboretum->begin(); pot_tree_it!=wrangler_pot.pot_arboretum->end(); pot_tree_it++){

    std::cout<<"Begin looping over "<<(*pot_tree_it)->old_pot_tree->GetName()<<" tree with "<<nentries<<" entries"<<std::endl;
    nentries = (*pot_tree_it)->old_pot_tree->GetEntries();
    for (Int_t i=0;i!=nentries;i++){

      if (i%10000 == 0) std::cout << i/1000 << " k " << std::setprecision(3) << double(i)/nentries*100. << " %"<< std::endl;

      (*pot_tree_it)->old_pot_tree->GetEntry(i);

      if (remove_set.find(std::make_pair((*pot_tree_it)->runNo, (*pot_tree_it)->subRunNo)) != remove_set.end()) continue;
      if (flag_data && skip_cut == 0){
        if (good_runlist_set.find((*pot_tree_it)->runNo) == good_runlist_set.end()) continue;
        if (low_lifetime_set.find((*pot_tree_it)->runNo) != low_lifetime_set.end()) continue;
        if (flag_numi && low_neutrino_count_numi_run2RHC_set.find((*pot_tree_it)->runNo) != low_neutrino_count_numi_run2RHC_set.end()) continue;
        //bad run in run 1 due to beam filter bnb
        //if (pot.runNo <= 5367 && pot.runNo >= 5320) continue;
        // ext bnb in run 1, high rate
        if ((*pot_tree_it)->runNo >=7004 && (*pot_tree_it)->runNo <=7070) continue;
        // ext bnb in run 2, high rate not in good list anyway
        //if ((pot.runNo>=10287 && pot.runNo <= 10304) || (pot.runNo>=12277 && pot.runNo <=12350)) continue;
        // ext bnb in run 2, low rate not in good list anyway
        //if ((pot.runNo>=9768 && pot.runNo <= 10070) || (pot.runNo>=10102 && pot.runNo <=10246)) continue;
        // bnb run 2 high rate
        if ((*pot_tree_it)->runNo >= 8321 && (*pot_tree_it)->runNo <=8404) continue;
        // bnb run 3 high rate
        if ((*pot_tree_it)->runNo >=15369 && (*pot_tree_it)->runNo <= 15402) continue;
      }
      (*pot_tree_it)->new_pot_tree->Fill();

    }//i, loop over events in a given pot tree set

  }//pot_trees_it, loop over sets of pot trees


  for (auto it = remove_set.begin(); it!= remove_set.end(); it++){
    std::cout <<"remove  run:" << it->first << " subrun:" << it->second << std::endl;
  }

  file2->Write("",TFile::kOverwrite);
  file2->Close();

  return 0;



}
