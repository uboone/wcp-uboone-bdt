#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "WCPLEEANA/tree_wrangler.h"

using namespace std;
using namespace LEEana;

#include "WCPLEEANA/pot.h"


// Truth-level keys used to identify duplicate copies of the same generated event.
// ExactKey requires bit-for-bit agreement; BucketKey quantizes the continuous
// fields (nuEnergy, vtxX/Y/Z) into bins of width truth_tolerance so that values
// separated only by floating-point noise from reprocessing still group together.
// (Values that straddle a bin edge by chance could in principle be missed, but
// for a tolerance this small relative to the physical spread of these variables
// that risk is negligible.)
using ExactKey  = std::tuple<Float_t, Int_t, Bool_t, Float_t, Float_t, Float_t>;
using BucketKey = std::tuple<long, Int_t, Bool_t, long, long, long>;
using RunSubrun = std::pair<int,int>;

long to_bucket(Float_t v, double tolerance){
  return std::lround(double(v) / tolerance);
}

// Fraction of the events originally in this (run,subrun) that survive after
// removing duplicates -- e.g. 1 removed out of 10 gives a pass_ratio of 0.9.
double get_pass_ratio(int run, int subrun,
                       const std::map<RunSubrun,int>& map_total,
                       const std::map<RunSubrun,int>& map_removed){
  RunSubrun key = std::make_pair(run,subrun);
  auto it_total = map_total.find(key);
  if (it_total == map_total.end() || it_total->second == 0) return 1.0;

  int removed = 0;
  auto it_removed = map_removed.find(key);
  if (it_removed != map_removed.end()) removed = it_removed->second;

  return 1.0 - double(removed) / double(it_total->second);
}


void print_help() {
  std::cout << R"(

========================================
 nuwro_duplicate_filter : Help
========================================

Overview:
---------
nuwro_duplicate_filter reads a ntuple file and removes duplicate
copies of the same generated (truth-level) event. 
The app keeps a single copy of each such event and rescales the POT 
of the affected subrun(s) to account for the events it removed.

The program:
  - Groups events in T_eval by
      (truth_nuEnergy, truth_nuPdg, truth_isCC, truth_vtxX, truth_vtxY, truth_vtxZ)
  - For any group with more than one entry, keeps the first (lowest entry
    index) copy and removes the rest
  - Rescales POT per subrun by the fraction of events kept in that subrun
  - Writes a clean output file with consistent POT accounting
  - Optionally copies additional trees via config file (see -H)

Usage:
------
  nuwro_duplicate_filter <input_file> <output_file> [options]

Required arguments:
-------------------
  input_file     Input ROOT file (Wire-Cell format, MC only)
  output_file    Output ROOT file

Options:
--------

  -h
      Show this help message and exit

  -H
      Show help message for configuration file and exit

  -t<string>
      Configuration file for tree selection
      (default: config.txt)

  -d<char>
      Delimiter used in config file
      (default: ',')

  -e<float>
      Tolerance used when comparing truth_nuEnergy, truth_vtxX, truth_vtxY
      and truth_vtxZ (truth_nuPdg and truth_isCC always require an exact
      match). Two entries are grouped as duplicates if each of these four
      values agrees to within this tolerance.
        > 0  = tolerance match (default: 0.01)
        0    = require bit-for-bit exact match on all six truth fields

Processing Details:
-------------------

  Duplicate Identification:
    Two entries are considered duplicates if their truth_nuPdg and
    truth_isCC match exactly, and their truth_nuEnergy, truth_vtxX,
    truth_vtxY and truth_vtxZ each agree to within the -e tolerance
    (default 0.01). Matching is not restricted to a single subrun.
    A small default tolerance is used because reprocessing can leave
    tiny (sub-0.01) floating-point differences between two copies of
    what is otherwise clearly the same generated event; -e0 disables
    this and requires an exact match instead.

  POT Handling:
    - POT is tracked per subrun, both for the Wire-Cell T_pot tree and for
      any other POT trees declared in the config file (exclusive/POT
      section), using the tree_wrangler POT arboretum.
    - Each subrun's POT is scaled by (events kept / events originally in
      that subrun), i.e. removing 1 of 10 events scales POT to 90%.
    - Output includes a 'pass_ratio' branch on every POT tree.

  Requires MC:
    Truth branches must be present on T_eval. The program exits with an
    error if they are not found (e.g. if run on real data).

Configuration File:
-------------------
Controls which additional trees are copied and how they are filtered.
Run:
  nuwro_duplicate_filter -H
for full details.

If no config file is used:
  Only Wire-Cell trees under 'wcpselection' are written.

)";
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
  else if (argc < 3) {
    std::cout << "nuwro_duplicate_filter #input_file #output_file " << std::endl;
    std::cout << "nuwro_duplicate_filter -h for further help and instructions." << std::endl;
    return -1;
  }

  TString input_file = argv[1];
  TString out_file = argv[2];

  bool flag_config = false;
  std::string config_file_name="config.txt";

  char delimiter = ',';

  double truth_tolerance = 0.01;

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
    if (!value_ptr) {
      std::cerr << "Missing value for -" << flag << std::endl;
      continue;
    }

    switch(flag){

    case 't':
      config_file_name = value_ptr;
      flag_config = true;
      break;

    case 'd':
      delimiter = value_ptr[0];
      break;

    case 'e':
      truth_tolerance = atof(value_ptr);
      if (truth_tolerance < 0) truth_tolerance = 0;
      break;
    }

  }

  tree_wrangler wrangler(flag_config, config_file_name, delimiter);
  tree_wrangler wrangler_ex(flag_config, config_file_name, delimiter,2);
  tree_wrangler wrangler_pot(flag_config, config_file_name, delimiter,1);

  // Always load WC
  TFile *file1 = new TFile(input_file);
  TTree *T_BDTvars    = (TTree*)file1->Get("wcpselection/T_BDTvars");
  TTree *T_eval       = (TTree*)file1->Get("wcpselection/T_eval");
  TTree *T_pot        = (TTree*)file1->Get("wcpselection/T_pot");
  TTree *T_PFeval     = (TTree*)file1->Get("wcpselection/T_PFeval");
  TTree *T_KINEvars   = (TTree*)file1->Get("wcpselection/T_KINEvars");
  TTree *T_spacepoints= (TTree*)file1->Get("wcpselection/T_spacepoints");

  if (!T_eval->GetBranch("truth_nuEnergy") || !T_eval->GetBranch("truth_nuPdg") ||
      !T_eval->GetBranch("truth_isCC")     || !T_eval->GetBranch("truth_vtxX")  ||
      !T_eval->GetBranch("truth_vtxY")     || !T_eval->GetBranch("truth_vtxZ")) {
    std::cout<<"\nERROR: T_eval is missing truth branches. nuwro_duplicate_filter requires MC input with truth information. Exiting.\n"<<std::endl;
    return 1;
  }

  // Load other trees from directories as specified by the config file
  wrangler.get_old_trees(file1);
  wrangler_ex.get_old_trees(file1);
  wrangler_pot.get_old_trees(file1);

  Int_t run, subrun, event;
  Float_t truth_nuEnergy, truth_vtxX, truth_vtxY, truth_vtxZ;
  Int_t truth_nuPdg;
  Bool_t truth_isCC;

  T_eval->SetBranchAddress("run", &run);
  T_eval->SetBranchAddress("subrun", &subrun);
  T_eval->SetBranchAddress("event", &event);
  T_eval->SetBranchAddress("truth_nuEnergy", &truth_nuEnergy);
  T_eval->SetBranchAddress("truth_nuPdg", &truth_nuPdg);
  T_eval->SetBranchAddress("truth_isCC", &truth_isCC);
  T_eval->SetBranchAddress("truth_vtxX", &truth_vtxX);
  T_eval->SetBranchAddress("truth_vtxY", &truth_vtxY);
  T_eval->SetBranchAddress("truth_vtxZ", &truth_vtxZ);

  // ---- Pass 1: find duplicate truth records in T_eval ----

  std::map<RunSubrun, int> map_rs_total_events;

  // Groups of entry indices that share the same truth-level key. Within each
  // group indices are in ascending order, since we only ever append while
  // scanning i upward.
  std::vector<std::vector<int> > duplicate_candidate_groups;

  int nentries_eval = T_eval->GetEntries();

  if (truth_tolerance <= 0){
    std::cout<<"\nScanning "<<nentries_eval<<" events for duplicate truth records (exact match) ..."<<std::endl;
    std::map<ExactKey, std::vector<int> > map_truth_to_indices;
    for (int i=0; i!=nentries_eval; i++){
      T_eval->GetEntry(i);
      map_rs_total_events[std::make_pair(run,subrun)]++;
      ExactKey key = std::make_tuple(truth_nuEnergy, truth_nuPdg, truth_isCC, truth_vtxX, truth_vtxY, truth_vtxZ);
      map_truth_to_indices[key].push_back(i);
    }
    for (auto& kv : map_truth_to_indices) duplicate_candidate_groups.push_back(kv.second);
  }
  else {
    std::cout<<"\nScanning "<<nentries_eval<<" events for duplicate truth records (tolerance = "<<truth_tolerance<<") ..."<<std::endl;
    std::map<BucketKey, std::vector<int> > map_truth_to_indices;
    for (int i=0; i!=nentries_eval; i++){
      T_eval->GetEntry(i);
      map_rs_total_events[std::make_pair(run,subrun)]++;
      BucketKey key = std::make_tuple(to_bucket(truth_nuEnergy, truth_tolerance), truth_nuPdg, truth_isCC,
                                       to_bucket(truth_vtxX, truth_tolerance), to_bucket(truth_vtxY, truth_tolerance), to_bucket(truth_vtxZ, truth_tolerance));
      map_truth_to_indices[key].push_back(i);
    }
    for (auto& kv : map_truth_to_indices) duplicate_candidate_groups.push_back(kv.second);
  }

  std::set<int> remove_index_set;
  std::map<RunSubrun, int> map_rs_removed_events;
  int n_duplicate_groups = 0;

  for (auto& indices : duplicate_candidate_groups){
    if (indices.size() < 2) continue;
    n_duplicate_groups++;

    // Keep the first (lowest entry index) copy, remove the rest.
    int kept_index = indices.front();
    T_eval->GetEntry(kept_index);
    int kept_run = run, kept_subrun = subrun, kept_event = event;

    for (size_t k=1; k<indices.size(); k++){
      int idx = indices[k];
      T_eval->GetEntry(idx);
      std::cout<<"Removing duplicate: run,subrun,event = "<<run<<", "<<subrun<<", "<<event
               <<"  (matches kept run,subrun,event = "<<kept_run<<", "<<kept_subrun<<", "<<kept_event<<")"<<std::endl;
      remove_index_set.insert(idx);
      map_rs_removed_events[std::make_pair(run,subrun)]++;
    }
  }

  std::cout<<"\nFound "<<n_duplicate_groups<<" duplicate truth groups, removing "
           <<remove_index_set.size()<<" of "<<nentries_eval<<" events.\n"<<std::endl;

  // ---- Set up the output file ----

  TFile *file2 = new TFile(out_file,"RECREATE");

  // Setup the directories specified in the config file
  wrangler.set_new_trees(file2);
  wrangler_ex.set_new_trees(file2);
  wrangler_pot.set_new_trees(file2);

  // Build the pairs of pot trees
  wrangler_pot.grow_pot_arboretum();
  wrangler_pot.map_rs_to_entry();

  file2->mkdir("wcpselection");
  file2->cd("wcpselection");

  TTree *new_T_eval        = T_eval->CloneTree(0);
  TTree *new_T_BDTvars     = T_BDTvars->CloneTree(0);
  TTree *new_T_PFeval      = T_PFeval->CloneTree(0);
  TTree *new_T_KINEvars    = T_KINEvars->CloneTree(0);
  TTree *new_T_spacepoints = T_spacepoints->CloneTree(0);

  POTInfo pot;
  set_tree_address(T_pot, pot);
  TTree *new_T_pot = new TTree("T_pot","T_pot");
  put_tree_address(new_T_pot, pot);
  float pass_ratio;
  new_T_pot->Branch("pass_ratio",&pass_ratio,"pass_ratio/F");

  // ---- Pass 2: copy every event-level tree, skipping removed duplicates ----

  int ientry=0;
  std::cout<<"Begin looping over "<<nentries_eval<<" events to fill event-level trees."<<std::endl;
  for (int i=0;i!=nentries_eval;i++){

    if (i%10000 == 0) std::cout << i/1000 << " k " << std::setprecision(3) << double(i)/nentries_eval*100. << " %"<< std::endl;

    if (remove_index_set.find(i) != remove_index_set.end()) continue;

    T_eval->GetEntry(i);
    T_BDTvars->GetEntry(i);
    T_PFeval->GetEntry(i);
    T_KINEvars->GetEntry(i);
    T_spacepoints->GetEntry(i);

    new_T_eval->Fill();
    new_T_BDTvars->Fill();
    new_T_PFeval->Fill();
    new_T_KINEvars->Fill();
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

    ientry++;
  }
  std::cout << "    kept: "<<ientry<<"/"<<nentries_eval<<std::endl;

  // ---- POT trees: scale each subrun by the fraction of events kept ----

  int nentries_pot = T_pot->GetEntries();
  std::cout<<"\nBegin looping over WC pot tree with "<<nentries_pot<<" entries"<<std::endl;
  for (int i=0;i!=nentries_pot;i++){

    T_pot->GetEntry(i);

    pass_ratio = get_pass_ratio(pot.runNo, pot.subRunNo, map_rs_total_events, map_rs_removed_events);

    pot.pot_tor875 *= pass_ratio;
    pot.pot_tor875good *= pass_ratio;

    new_T_pot->Fill();
  }

  // Now the other (exclusive/POT section) trees, one per entry in the arboretum
  for(auto pot_tree_it=wrangler_pot.pot_arboretum->begin(); pot_tree_it!=wrangler_pot.pot_arboretum->end(); pot_tree_it++){

    (*pot_tree_it)->new_pot_tree->Branch("pass_ratio",&pass_ratio,"pass_ratio/F");

    int nentries_arb = (*pot_tree_it)->old_pot_tree->GetEntries();
    std::cout<<"Begin looping over "<<(*pot_tree_it)->old_pot_tree->GetName()<<" tree with "<<nentries_arb<<" entries"<<std::endl;

    for (int i=0;i!=nentries_arb;i++){

      (*pot_tree_it)->old_pot_tree->GetEntry(i);

      pass_ratio = get_pass_ratio((*pot_tree_it)->runNo, (*pot_tree_it)->subRunNo, map_rs_total_events, map_rs_removed_events);

      // Set both the double and the float, only the correct one will fill the tree
      (*pot_tree_it)->fpot = (*pot_tree_it)->pot() * pass_ratio;
      (*pot_tree_it)->dpot = (*pot_tree_it)->pot() * pass_ratio;

      (*pot_tree_it)->new_pot_tree->Fill();

    }

  }

  file2->Write("",TFile::kOverwrite);
  file2->Close();

  std::cout<<"\nEvents: "<<ientry<<"/"<<nentries_eval<<std::endl;
  std::cout<<"Duplicate truth groups found: "<<n_duplicate_groups<<", events removed: "<<remove_index_set.size()<<std::endl;
  std::cout<<"\nSaving output file: "<<out_file<<"\n"<<std::endl;

  return 0;
}
