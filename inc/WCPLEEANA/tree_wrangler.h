#ifndef LEEANAC_TREE_WRANGLER
#define LEEANAC_TREE_WRANGLER

#include "WCPLEEANA/Util.h"

#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TROOT.h"

#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>

namespace LEEana{

  struct pot_tree_pair{
    TTree* new_pot_tree;
    TTree* old_pot_tree;
    bool isfloat;
    int runNo;
    int subRunNo;
    float fpot;
    double dpot;
    double pot(){ //load both the double and the float, this then returns the one that we need
      if(isfloat) return fpot;
      else return dpot;
    };
  };

  class tree_wrangler{
  public:
    tree_wrangler(bool configure=true, std::string config_file_name="config.txt", char delimiter=',', int set_flag_exclusive=0, bool set_verbose=false);
    ~tree_wrangler();

    void get_old_trees(TFile* file);
    void set_new_trees(TFile* file, bool rename=false, TString TDirectory_extension="");

    void grow_pot_arboretum();

    void map_rs_to_entry(); 

    std::vector<TTree*>* new_trees;
    std::vector<TTree*>* old_trees;
    std::vector<pot_tree_pair*>* pot_arboretum;	
    std::vector<std::map<std::pair<int, int>, std::pair<int, double> > > arboretum_map_rs_entry; 

    void set_samdef(bool f_flag_save_samdef, TString f_samdef);

    std::map<std::string, std::tuple<TDirectory*,std::vector<TTree*>*>> names_wi_directories_and_trees;
 
  private:
    bool verbose;
    int flag_exclusive;
    std::map<std::string,std::vector<std::string>> directories_wi_trees_to_skip_names;
    std::map<std::string,std::vector<std::string>> directories_wi_pot_var_names;
    std::map<std::string,std::vector<std::string>> trees_wi_pot_var_names;
    std::vector<std::vector<std::string>> pot_var_names;

    void CopyDir(TDirectory *source, bool blank_tree=false, std::vector<std::string> to_skip={});
    void CopyDir(TDirectory *source, TString TDirectory_extension, bool blank_tree=false, std::vector<std::string> to_skip={});
    
    std::vector<TTree*>* CopyTrees(TDirectory *source, bool blank_tree=false, bool rename=false, TString TDirectory_extension="", std::vector<std::string> to_skip={});
    std::vector<TTree*>* GetTrees(TDirectory *source, std::vector<std::string> to_skip={});

    bool flag_save_samdef = false;
    TString samdef = "";
  };

  // tree with the run/subrun/event of each entry; only these branches are left enabled, so pass a TFile of its own
  // (not the one the wrangler copies from)
  int get_T_rse(TFile* file1, TTree*& T_rse, int &run, int &subrun, int &event);

  std::vector<int> get_low_lifetime_runs();
  std::vector<int> get_good_run_list();
  std::vector<int> get_low_neutrino_count_numi_run2RHC();

  void print_help_wrangler_config(bool pick=true);
}

#endif
