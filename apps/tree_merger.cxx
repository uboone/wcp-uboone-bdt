#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <set>
#include <limits>
#include <unordered_map>
#include <vector>
#include <tuple>

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


using TreeKey = std::pair<std::string,std::string>; // (dir, tree)

std::map<TreeKey, TTree*> build_tree_map(tree_wrangler& w)
{
  std::map<TreeKey, TTree*> out;

  for (const auto& kv : w.names_wi_directories_and_trees) {
    const std::string& dir = kv.first;
    auto trees = std::get<1>(kv.second);

    for (auto* t : *trees) {
      out[std::make_pair(dir, t->GetName())] = t;
    }
  }

  return out;
}


void build_output_trees(tree_wrangler& w1, tree_wrangler& w2, std::vector<TTree*>* new_trees, TFile* out_file){

  auto map1 = build_tree_map(w1);
  auto map2 = build_tree_map(w2);

  std::set<TreeKey> all_keys;
  for (auto& kv : map1) all_keys.insert(kv.first);
  for (auto& kv : map2) all_keys.insert(kv.first);

  for (const auto& key : all_keys) {

    const std::string& dir  = key.first;
    const std::string& name = key.second;

    TTree* t1 = map1.count(key) ? map1[key] : nullptr;
    TTree* t2 = map2.count(key) ? map2[key] : nullptr;

    // Ensure directory exists and go there
    if (!out_file->GetDirectory(dir.c_str())){
      out_file->mkdir(dir.c_str());
    }
    out_file->cd(dir.c_str());

    TTree* out_tree = nullptr;

    // Get the trees only in one file or another
    if (t1 && !t2) {
      out_tree = t1->CloneTree(0);
    } else if (t2 && !t1) {
      out_tree = t2->CloneTree(0);
    }

    // Merge branches from t2 into t1 if both exist
    if (t1 && t2) {

      // Enable all branches
      t1->SetBranchStatus("*",1);
      t2->SetBranchStatus("*",1);

      // Clone t1
      out_tree = t1->CloneTree(0);

      // Now add t2 branches safely
      TObjArray* b2 = t2->GetListOfBranches();
      for (int i=0;i<b2->GetEntries();i++) {
        TBranch* br = (TBranch*)b2->At(i);
        std::string name = br->GetName();
        if (out_tree->GetBranch(name.c_str())) continue;
        t2->SetBranchAddress(name.c_str(), br->GetAddress());
        out_tree->Branch(name.c_str(), br->GetAddress(), br->GetTitle());
      }

    }

    out_tree->Write("",TTree::kOverwrite);
    new_trees->push_back(out_tree);

  }

}


void print_help() {
  std::cout << R"(

========================================
 tree_merger : ROOT TTree merging tool
========================================

USAGE:
  tree_merger <input_file1> <input_file2> <output_file> <config1.txt> <config2.txt> [options]

DESCRIPTION:
  Merges TTrees from two ROOT files based on matching (run, subrun, event).
  Trees are matched and combined according to configuration files.
  Output file will contain merged trees and copied POT trees.
  This is usefull in recovering or adding branches or trees to a file which has been trimmed.
  It can also be used to add new custom variables without needing to re-do other proscessing.

REQUIRED ARGUMENTS:
  input_file1     First ROOT input file (reference file)
  input_file2     Second ROOT input file (must contain matching or superset of events, need -m1 for the latter)
  output_file     Output ROOT file to write merged trees
  config1.txt     Configuration file for input_file1, tell what trees and banches will be in the output file
  config2.txt     Configuration file for input_file2, tell what trees and banches will be in the output file

OPTIONS:
  -h              Print this help message and exit
  -H              Print extended configuration help
  -d<char>        Set delimiter used in config files (default: ',')
  -o<int>         Overwrite output file if it exists (1 = overwrite, default: 0)
  -m<int>         Mismatch handling mode:
                    0 = Fail if event counts differ (default)
                    1 = Allow file2 to have extra events (they will be dropped)
  -v<int>         Verbosity for event loop printing (default: 10000)
  -u<int>         Verbosity for POT loop printing (default: 1000)

BEHAVIOR NOTES:
  - file1 is treated as the reference dataset.
  - file2 must contain the same events present in file1
    - Using -m1 allows file2 to have more events, but file1 must still be a subset of file2.
  - Trees present in only one file are copied directly.
  - Trees present in both files are merged branch-wise.
  - POT trees are copied with priority given to file1.

Configuration File:
run tree_trimmer -H for more info

EXAMPLES:
  tree_merger file1.root file2.root output.root cfg1.txt cfg2.txt
  tree_merger file1.root file2.root output.root cfg1.txt cfg2.txt -o1 -m1


========================================

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
  else if (argc < 6) {
    std::cout << "tree_merger #input_file1 #input_file2 #output_file #config1.txt #config2.txt" << std::endl;
    std::cout << "tree_merger -h for further help and instructions." << std::endl;
    return -1;
  }


  TString input_file1 = argv[1];
  TString input_file2 = argv[2];
  TString out_file = argv[3];

  std::string config_file1_name=argv[4];
  std::string config_file2_name=argv[5];

  char delimiter = ',';

  int flag_overwrite=0;

  int flag_mismatch_events=0;

  int set_verbose=10000;
  int set_verbose_pot=1000;

  for (Int_t i = 4; i < argc; ++i) {

    // Skip anything that isn't a flag
    if (argv[i][0] != '-') continue;

    char flag = argv[i][1];
    char* value_ptr = nullptr;

    // Case 1: value is attached (e.g. -o1)
    if (argv[i][2] != '\0') {
      value_ptr = &argv[i][2];
    }
    // Case 2: value is in next argument (e.g. -o 1)
    else if (i + 1 < argc) {
      value_ptr = argv[i + 1];
      ++i; // consume next argument
    }

    if (!value_ptr) {
      std::cerr << "WARNING: Missing value for -" << flag << std::endl;
    }

    switch(flag) {

    case 'd':
      if (value_ptr) delimiter = value_ptr[0];
      break;

    case 'o':
      if (value_ptr) flag_overwrite = atoi(value_ptr);
      break;

    case 'm':
      if (value_ptr) {
        flag_mismatch_events = atoi(value_ptr);

        if(flag_mismatch_events == 0){
          std::cout << "Fail if file1 and file2 have a different number of events.\n\n";
        }
        else if(flag_mismatch_events == 1){
          std::cout << "Try to drop events from file2 if it has more events than file1\n\n";
        }
        else{
          std::cout << "Unknown -m option, setting to default flag_mismatch_events=0\n\n";
          flag_mismatch_events = 0;
        }
    }
    break;

    case 'v':
      if (value_ptr && atoi(value_ptr) > 0) {
        set_verbose = atoi(value_ptr);
      } else {
        std::cout << "Bad -v option, must be greater than 0. Leaving at 10000.\n\n";
      }
      break;

    case 'u':
      if (value_ptr && atoi(value_ptr) > 0) {
        set_verbose_pot = atoi(value_ptr);
      } else {
        std::cout << "Bad -u option, must be greater than 0. Leaving at 1000.\n\n";
      }
      break;

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


  // Initiate the tree wranglers
  tree_wrangler wrangler1(true, config_file1_name, delimiter);
  tree_wrangler wrangler_ex1(true, config_file1_name, delimiter,2);
  tree_wrangler wrangler_pot1(true, config_file1_name, delimiter,1);

  tree_wrangler wrangler2(true, config_file2_name, delimiter);
  tree_wrangler wrangler_ex2(true, config_file2_name, delimiter,2);
  tree_wrangler wrangler_pot2(true, config_file2_name, delimiter,1);


  TFile *file1 = new TFile(input_file1);
  TFile *file2 = new TFile(input_file2);


  // Load other trees from directories as specified by the config file
  wrangler1.get_old_trees(file1);
  wrangler_ex1.get_old_trees(file1);
  wrangler_pot1.get_old_trees(file1);
  wrangler2.get_old_trees(file2);
  wrangler_ex2.get_old_trees(file2);
  wrangler_pot2.get_old_trees(file2);


  // Build the pairs of pot trees, only need this for file2
  wrangler_pot2.new_trees = wrangler_pot2.old_trees; 
  wrangler_pot2.grow_pot_arboretum();


  // Figure out which tree we can load RSE from for file 1
  int run1;
  int subrun1;
  int event1;
  TTree *T_rse1 = nullptr;
  // The first passes read run/subrun/event from second handles of the input files, with only these branches enabled
  // (the trees of file1 and file2 are the ones being copied)
  TFile *file1_rse = new TFile(input_file1);
  TFile *file2_rse = new TFile(input_file2);
  int found_rse_tree1 = get_T_rse(file1_rse, T_rse1, run1, subrun1, event1);
  if(!found_rse_tree1){
    std::cout<<'\n'<<"Could not find RSE tree for file1. Exiting."<<std::endl;
    return 1;
  }

  // Figure out which tree we can load RSE from for file2
  int run2;
  int subrun2;
  int event2;
  TTree *T_rse2 = nullptr;
  int found_rse_tree2 = get_T_rse(file2_rse, T_rse2, run2, subrun2, event2);
  if(!found_rse_tree2){
    std::cout<<'\n'<<"Could not find RSE tree for file2. Exiting."<<std::endl;
    return 1;
  }

  // Check entry numbers make sense.
  int nentry1 = T_rse1->GetEntries();
  int nentry2 = T_rse2->GetEntries();
  if(nentry1!=nentry2){
    std::cout<<"Number of events in each file mismatch. Exiting."<<std::endl;
    if(flag_mismatch_events==0){
      std::cout<<"Exiting. You can overide this behavior with -m1"<<std::endl;
      return 1;
    }
    if(flag_mismatch_events==1 && nentry1<nentry2){
      std::cout<<"OK: file2 has more events than file1, will drop file2 events not in file1."<<std::endl;
      std::cout<<"The program will still fail if file1 is not a subset of file2."<<std::endl;
    }
    else{
      std::cout<<"BAD: file1 has more events than file2, but file1 must be a subset of file2."<<std::endl;
      std::cout<<"Try switching the file order if you expect this is the case."<<std::endl;
      std::cout<<"Exiting"<<std::endl;
      return 1;
    }
  }


  TFile *fileout = new TFile(out_file,"RECREATE");

  std::vector<TTree*>* new_trees = new std::vector<TTree*>;

  build_output_trees(wrangler1, wrangler2, new_trees, fileout);
  build_output_trees(wrangler_ex1, wrangler_ex2, new_trees, fileout);


  // Build RSE map for file 1
  // Map out the relation between run -> subrun -> (event,index) for file 2
  int verbose_counter=0;
  int index_counter=0;
  int event_counter=0;
  std::cout<<"Starting first pass loop over file1. Will pass over "<<nentry1<<" entries."<<std::endl;
  std::unordered_map<int, std::unordered_map<int, std::unordered_map<int,int > > > run_subrun_event_index_file1;
  std::vector<std::tuple<int,int,int> > rse_file1(nentry1);
  for (Int_t i=0;i!=nentry1;i++){
    if ((i-verbose_counter)%set_verbose == 0) {
      std::cout << "    seen: "<<i<<"    passed: "  << event_counter << std::endl;
      verbose_counter=(int(int(i)/int(set_verbose)))*set_verbose;
    }
    T_rse1->GetEntry(i);
    run_subrun_event_index_file1[run1][subrun1][event1] = i;
    rse_file1.at(i) = std::make_tuple(run1, subrun1, event1);
    index_counter=index_counter+1;
    event_counter=event_counter+1;
  }
  std::cout << "    seen: "<<nentry1<<"    passed: "  << event_counter << std::endl;

  // Build RSE map for file 2
  // Map out the relation between run -> subrun -> (event,index) for file 2
  verbose_counter=0;
  index_counter=0;
  event_counter=0;
  std::cout<<"Starting first pass loop over file2. Will pass over "<<nentry2<<" entries."<<std::endl;
  std::unordered_map<int, std::unordered_map<int, std::unordered_map<int,int > > > run_subrun_event_index_file2;
  for (Int_t i=0;i!=nentry2;i++){
    if ((i-verbose_counter)%set_verbose == 0) {
      std::cout << "    seen: "<<i<<"    passed: "  << event_counter << std::endl;
      verbose_counter=(int(int(i)/int(set_verbose)))*set_verbose;
    }
    T_rse2->GetEntry(i);
    run_subrun_event_index_file2[run2][subrun2][event2] = i;
    index_counter=index_counter+1;
    event_counter=event_counter+1;
  }
  std::cout << "    seen: "<<nentry2<<"    passed: "  << event_counter << std::endl;

  //fileout->cd();


  // Loop over file 1 and do the filling

  std::cout<<'\n'<<'\n'<<"Begin looping over "<<nentry1<<" events to fill trees."<<std::endl;
  verbose_counter=0;
  index_counter=0;
  event_counter=0;

  // Loop over file1 in the order of its entries (file2 follows it when the files have the same order): reading in the
  // order of the hash maps jumps through the files and decompresses a new basket for nearly every event.
  for (Int_t this_index1=0; this_index1!=nentry1; this_index1++){

    int this_run = std::get<0>(rse_file1.at(this_index1));
    int this_subrun = std::get<1>(rse_file1.at(this_index1));
    int this_event = std::get<2>(rse_file1.at(this_index1));

    // Each run-subrun-event once, as in the map (the last entry of a duplicated event)
    if (run_subrun_event_index_file1[this_run][this_subrun][this_event] != this_index1) continue;

    if (run_subrun_event_index_file2.find(this_run) == run_subrun_event_index_file2.end()) {
      std::cout<<"file 2 is missing run = "<<this_run<<". Exiting."<<std::endl;
      return 1;
    }
    if (run_subrun_event_index_file2[this_run].find(this_subrun) == run_subrun_event_index_file2[this_run].end()) {
      std::cout<<"file 2 is missing run,subrun = "<<this_run<<", "<<this_subrun<<". Exiting."<<std::endl;
      return 1;
    }
    if (run_subrun_event_index_file2[this_run][this_subrun].find(this_event) == run_subrun_event_index_file2[this_run][this_subrun].end()) {
      std::cout<<"file 2 is missing run,subrun,event = "<<this_run<<", "<<this_subrun<<", "<<this_event<<". Exiting."<<std::endl;
      return 1;
    }

    int this_index2 = run_subrun_event_index_file2[this_run][this_subrun][this_event];

    // Now fill all the trees.
    for(auto tree_it=wrangler1.old_trees->begin(); tree_it!=wrangler1.old_trees->end(); tree_it++){
      (*tree_it)->GetEntry(this_index1);
    }
    for(auto tree_it=wrangler2.old_trees->begin(); tree_it!=wrangler2.old_trees->end(); tree_it++){
      (*tree_it)->GetEntry(this_index2);
    }
    for(auto tree_it=wrangler_ex1.old_trees->begin(); tree_it!=wrangler_ex1.old_trees->end(); tree_it++){
      (*tree_it)->GetEntry(this_index1);
    }
    for(auto tree_it=wrangler_ex2.old_trees->begin(); tree_it!=wrangler_ex2.old_trees->end(); tree_it++){
      (*tree_it)->GetEntry(this_index2);
    }
    for(auto tree_it=new_trees->begin(); tree_it!=new_trees->end(); tree_it++){
      (*tree_it)->Fill();
    }

    if ((index_counter-verbose_counter)%set_verbose == 0) {
      std::cout << "    seen: "<<index_counter<<"    saved: "  << event_counter<< std::endl;
      verbose_counter=(int(int(index_counter)/int(set_verbose)))*set_verbose;
    }
    index_counter=index_counter+1;
    event_counter=event_counter+1;

  } // this_index1
  std::cout << "    seen: "<<index_counter<<"    saved: "  << event_counter<< std::endl;


  // Directly copy POT trees from file1, then file2.
  // Prioritize file1, if both trees contain a tree, it is taken from file1

  std::cout<<"\n"<<"\n"<<"Begin saving POT trees"<<std::endl;

  auto map1 = build_tree_map(wrangler_pot1);
  auto map2 = build_tree_map(wrangler_pot2);

  std::set<TreeKey> all_keys;
  for (auto& kv : map1) all_keys.insert(kv.first);
  for (auto& kv : map2) all_keys.insert(kv.first);

  for (const auto& key : all_keys) {

    const std::string& dir  = key.first;
    const std::string& name = key.second;

    TTree* t1 = map1.count(key) ? map1[key] : nullptr;
    TTree* t2 = map2.count(key) ? map2[key] : nullptr;

    // Ensure directory exists and go there
    if (!fileout->GetDirectory(dir.c_str())){
      fileout->mkdir(dir.c_str());
    }
    fileout->cd(dir.c_str());

    TTree* out_tree = nullptr;

    // Get the trees only in one file or another
    if (t1) {
      out_tree = t1->CloneTree(-1);
    } else if (t2 && nentry1==nentry2) {
      out_tree = t2->CloneTree(-1);
    } else if (t2) {

      // Now we gotta get fancy to get the POT correct becouse file2 can have more entries than file1
      std::cout<<"\n"<<"\n"<<"Attempting to cull pot tree named "<<name<<std::endl;

      out_tree = t2->CloneTree(0);

      auto arb_pot_tree_it=wrangler_pot2.pot_arboretum->begin();
      // find corresponding tree in the pot arboretum
      for(auto pot_tree_it=wrangler_pot2.pot_arboretum->begin(); pot_tree_it!=wrangler_pot2.pot_arboretum->end(); pot_tree_it++){
        std::string this_pot_tree_name = (*pot_tree_it)->old_pot_tree->GetName();
        if(this_pot_tree_name==t2->GetName()){
          arb_pot_tree_it=pot_tree_it;
          break;
        }
      }  

      // First pass loop to map out RS
      verbose_counter=0;
      index_counter=0;
      event_counter=0;
      int nentry_pot = t2->GetEntries();
      std::cout<<'\n'<<"Starting first pass loop over "<<name<<" in file2. Will pass over "<<nentry_pot<<" entries."<<std::endl;
      std::unordered_map<int, std::unordered_map<int, int > > run_subrun_index_file2;
      for (Int_t i=0;i!=nentry_pot;i++){
        if ((i-verbose_counter)%set_verbose == 0) {
          std::cout << "    seen: "<<i<<"    passed: "  << event_counter << std::endl;
          verbose_counter=(int(int(i)/int(set_verbose_pot)))*set_verbose_pot;
        }
        t2->GetEntry(i);
        int this_run = (*arb_pot_tree_it)->runNo;
        int this_subrun = (*arb_pot_tree_it)->subRunNo;
        run_subrun_index_file2[this_run][this_subrun] = i;
        index_counter=index_counter+1;
        event_counter=event_counter+1;
      }
      std::cout << "    seen: "<<nentry_pot<<"    passed: "  << event_counter << std::endl;

      // Second pass loop to fill the output trees

      std::cout<<"Begin looping over "<<name<<" with "<<nentry_pot<<" events to fill POT tree only in file2"<<std::endl;
      verbose_counter=0;
      index_counter=0;
      event_counter=0;

      for (auto r_it = run_subrun_event_index_file1.begin(); r_it != run_subrun_event_index_file1.end(); r_it++){

        int this_run = (*r_it).first;
        if (run_subrun_event_index_file2.find(this_run) == run_subrun_event_index_file2.end()) {
          std::cout<<"file 2 is missing run = "<<this_run<<". Exiting."<<std::endl;
          return 1;
        }

        for (auto s_it = (*r_it).second.begin(); s_it != (*r_it).second.end(); s_it++){

          int this_subrun = (*s_it).first;
          if (run_subrun_event_index_file2[this_run].find(this_subrun) == run_subrun_event_index_file2[this_run].end()) {
            std::cout<<"file 2 is missing run,subrun = "<<this_run<<", "<<this_subrun<<". Exiting."<<std::endl;
            return 1;
          }

          int this_index = run_subrun_index_file2[this_run][this_subrun];

          // Now fill all the trees.
          t2->GetEntry(this_index);
          out_tree->Fill();

          if ((index_counter-verbose_counter)%set_verbose_pot == 0) {
            std::cout << "    seen: "<<index_counter<<"    saved: "  << event_counter<< std::endl;
            verbose_counter=(int(int(index_counter)/int(set_verbose_pot)))*set_verbose_pot;
          }
          index_counter=index_counter+1;
          event_counter=event_counter+1;

        } // s_it

      } // r_it
      std::cout << "    seen: "<<index_counter<<"    saved: "  << event_counter<< std::endl;

    }

    out_tree->Write("",TTree::kOverwrite);

  }


  fileout->Write("",TFile::kOverwrite);
  fileout->Close();


  std::cout<<'\n'<<'\n'<<"Saving output file: "<<out_file<<'\n'<<std::endl;


  return 0;

}
