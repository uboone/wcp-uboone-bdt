#ifndef UBOONE_LEE_PARTICLE
#define UBOONE_LEE_PARTICLE

#include <algorithm>
#include <numeric>
#include "WCPLEEANA/pfeval.h"
#include "WCPLEEANA/space.h"

namespace LEEana{
struct ParticleInfo{
    std::vector<float> *spacepoints_x;
    std::vector<float> *spacepoints_y;
    std::vector<float> *spacepoints_z;
    std::vector<float> *spacepoints_q;

    float spacepoints_q_0;
    float spacepoints_q_1;
    float spacepoints_q_2;
    float spacepoints_q_3;
    float spacepoints_q_4;
    float spacepoints_q_5;
    float spacepoints_q_6;
    float spacepoints_q_7;
    float spacepoints_q_8;
    float spacepoints_q_9;
    float spacepoints_q_10;
    float spacepoints_q_11;
    float spacepoints_q_12;
    float spacepoints_q_13;
    float spacepoints_q_14;
    float spacepoints_q_15;
    float spacepoints_q_16;
    float spacepoints_q_17;
    float spacepoints_q_18;
    float spacepoints_q_19;
    float spacepoints_q_20;
    float spacepoints_q_21;
    float spacepoints_q_22;
    float spacepoints_q_23;
    float spacepoints_q_24;

    float spacepoints_q_bck_0;
    float spacepoints_q_bck_1;
    float spacepoints_q_bck_2;
    float spacepoints_q_bck_3;
    float spacepoints_q_bck_4;
    float spacepoints_q_bck_5;
    float spacepoints_q_bck_6;
    float spacepoints_q_bck_7;
    float spacepoints_q_bck_8;
    float spacepoints_q_bck_9;
    float spacepoints_q_bck_10;
    float spacepoints_q_bck_11;
    float spacepoints_q_bck_12;
    float spacepoints_q_bck_13;
    float spacepoints_q_bck_14;
    float spacepoints_q_bck_15;
    float spacepoints_q_bck_16;
    float spacepoints_q_bck_17;
    float spacepoints_q_bck_18;
    float spacepoints_q_bck_19;
    float spacepoints_q_bck_20;
    float spacepoints_q_bck_21;
    float spacepoints_q_bck_22;
    float spacepoints_q_bck_23;
    float spacepoints_q_bck_24;

    float spacepoints_q_med;

    float flag_prim_mu;

    float flag_is_contained;

    float flag_has_daught;
    float flag_has_daught_p;
    float flag_has_daught_el;
    float flag_has_daught_pi;

    float reco_truthMatch_pdg;
    float reco_truthMatch_id;
    float reco_truthMatch_mother;
    float reco_truthMatch_energy;

    float reco_momentum_0;
    float reco_momentum_1;
    float reco_momentum_2;
    float reco_momentum_3;
    float reco_pdg;

    float reco_larpid_pdg;
    float reco_larpid_pidScore_el;
    float reco_larpid_pidScore_ph;
    float reco_larpid_pidScore_mu;
    float reco_larpid_pidScore_pr;
    float reco_larpid_pidScore_pi;
    float reco_larpid_proccess;

    float true_is_n_induced;
    float reco_is_n_induced;
    float reco_is_g_induced;

    float dist_to_vtx;
    float cos_theta;
    float proximity;

    float track_len;
    float direct_track_len;
    float track_len_ratio;
};

void create_particle(SpaceInfo& space_info, PFevalInfo& pfeval, ParticleInfo& particle_info, int index, bool flag_data, double tolerance_sp=0.5, double tolerance=0.0001);
void reset_particle(ParticleInfo& particle_info);
void print_particle(ParticleInfo& particle_info);
}


void LEEana::create_particle(SpaceInfo& space_info, PFevalInfo& pfeval, ParticleInfo& particle_info, int index, bool flag_data, double tolerance_sp, double tolerance){

  //std::cout<<"Clearing particle"<<std::endl;
  reset_particle(particle_info);

  //std::cout<<"Searching for particle"<<std::endl;
  // Loop over all particles in this event 
  for(int reco_part=0; reco_part<pfeval.reco_Ntrack; reco_part++){

    // Find the one associated with this id, skip if its a pseudo particle
    if(pfeval.reco_id[reco_part]!=pfeval.reco_id[index]) continue;
    //std::cout<<"Found ID"<<std::endl;
    if(pfeval.reco_pdg[reco_part]==2112 || pfeval.reco_pdg[reco_part]==22 || pfeval.reco_pdg[reco_part]==111) continue;
    //if(pfeval.reco_truthMatch_pdg[reco_part]<=0 && !flag_data) continue;
    //std::cout<<"Real particle and found amtch"<<std::endl;

    double part_x = pfeval.reco_startXYZT[reco_part][0];
    double part_y = pfeval.reco_startXYZT[reco_part][1];
    double part_z = pfeval.reco_startXYZT[reco_part][2];
    double part_end_x = pfeval.reco_endXYZT[reco_part][0];
    double part_end_y = pfeval.reco_endXYZT[reco_part][1];
    double part_end_z = pfeval.reco_endXYZT[reco_part][2];

    //std::cout<<"Setting spacepoints"<<std::endl;
    // Save the spacepoints with the same id as the current particle
    std::vector<float> temp_spacepoints_x;
    std::vector<float> temp_spacepoints_y;
    std::vector<float> temp_spacepoints_z;
    std::vector<float> temp_spacepoints_q;
    //std::cout<<"Number of spacepoints: "<<space_info.Trecchargeblob_spacepoints_real_cluster_id->size()<<std::endl;    
    for(int sp=0; sp<space_info.Trecchargeblob_spacepoints_real_cluster_id->size(); sp++){
      //std::cout<<"Checking sp "<<sp<<std::endl; 
      if(space_info.Trecchargeblob_spacepoints_real_cluster_id->at(sp)==pfeval.reco_id[reco_part]){
        //std::cout<<"Adding sp "<<sp<<" x="<<space_info.Trecchargeblob_spacepoints_x->at(sp)<<" q="<<space_info.Trecchargeblob_spacepoints_q->at(sp)<<std::endl; 
        temp_spacepoints_x.push_back(space_info.Trecchargeblob_spacepoints_x->at(sp));
        temp_spacepoints_y.push_back(space_info.Trecchargeblob_spacepoints_y->at(sp));
        temp_spacepoints_z.push_back(space_info.Trecchargeblob_spacepoints_z->at(sp));
        temp_spacepoints_q.push_back(space_info.Trecchargeblob_spacepoints_q->at(sp));
      }
    }
    int n_spacepoints = temp_spacepoints_x.size();
    if(n_spacepoints==0) continue;

    //std::cout<<"Computing length"<<std::endl;
    //Get the length of the proton by adding up the distance between each pair of spacepoints
    particle_info.track_len=0;
    for(int sp=0; sp<n_spacepoints-1; sp++){
      double dx = temp_spacepoints_x.at(sp) - temp_spacepoints_x.at(sp+1);
      double dy = temp_spacepoints_y.at(sp) - temp_spacepoints_y.at(sp+1);
      double dz = temp_spacepoints_z.at(sp) - temp_spacepoints_z.at(sp+1);
      double dist = sqrt(pow(dx,2)+pow(dy,2)+pow(dz,2));
      particle_info.track_len+=dist;
    }


    // Check the start and end of the proton track to see if one matches the first spacepoint. 
    // This tells us if the spacepoints for this track were saved track start-to-end or track end-to-start
    // If neither matches, throw an error, I think this can happen sometimes for mouns (vertex sometimes gets redefined in the reco)? But have not seen if for protons
    if( !(temp_spacepoints_x.at(0)>part_x-tolerance_sp && temp_spacepoints_x.at(0)<part_x+tolerance_sp && temp_spacepoints_y.at(0)>part_y-tolerance_sp && temp_spacepoints_y.at(0)<part_y+tolerance_sp && temp_spacepoints_z.at(0)>part_z-tolerance_sp && temp_spacepoints_z.at(0)<part_z+tolerance_sp) ){
      if(!(temp_spacepoints_x.back()>part_x-tolerance_sp && temp_spacepoints_x.back()<part_x+tolerance_sp && temp_spacepoints_y.back()>part_y-tolerance_sp && temp_spacepoints_y.back()<part_y+tolerance_sp && temp_spacepoints_z.back()>part_z-tolerance_sp && temp_spacepoints_z.back()<part_z+tolerance_sp) ){
        *particle_info.spacepoints_x = temp_spacepoints_x;
        *particle_info.spacepoints_y = temp_spacepoints_y;
        *particle_info.spacepoints_z = temp_spacepoints_z;
        *particle_info.spacepoints_q = temp_spacepoints_q; 
      }else{
        std::vector<float> rev_temp_spacepoints_x;
        std::vector<float> rev_temp_spacepoints_y;
        std::vector<float> rev_temp_spacepoints_z;
        std::vector<float> rev_temp_spacepoints_q;
        for(int sp=0; sp<n_spacepoints; sp++){
          rev_temp_spacepoints_x.push_back(temp_spacepoints_x.at(n_spacepoints-1-sp));
          rev_temp_spacepoints_y.push_back(temp_spacepoints_y.at(n_spacepoints-1-sp));
          rev_temp_spacepoints_z.push_back(temp_spacepoints_z.at(n_spacepoints-1-sp));
          rev_temp_spacepoints_q.push_back(temp_spacepoints_q.at(n_spacepoints-1-sp));
        }
        *particle_info.spacepoints_x = rev_temp_spacepoints_x;
        *particle_info.spacepoints_y = rev_temp_spacepoints_y;
        *particle_info.spacepoints_z = rev_temp_spacepoints_z;
        *particle_info.spacepoints_q = rev_temp_spacepoints_q;
      }
    }else{ 
      *particle_info.spacepoints_x = temp_spacepoints_x;
      *particle_info.spacepoints_y = temp_spacepoints_y;
      *particle_info.spacepoints_z = temp_spacepoints_z;
      *particle_info.spacepoints_q = temp_spacepoints_q;
    }

    // Save the median dqdx
    std::vector<float> sorted_spacepoints_q = *(particle_info.spacepoints_q);
    std::sort(sorted_spacepoints_q.begin(), sorted_spacepoints_q.end());
    size_t size = sorted_spacepoints_q.size();
    if (size % 2 == 0) {
      particle_info.spacepoints_q_med = (((sorted_spacepoints_q.at(size / 2 - 1) + sorted_spacepoints_q.at(size / 2)) / 2.0)+10000)*10;
    }else{
      particle_info.spacepoints_q_med  = ((sorted_spacepoints_q.at(size / 2))+10000)*10;
    }

    // Find the mother that was larpid matched and add some extra info on it
    int temp_truth_mother_id=-1;
    for(int truth_part=0; truth_part<pfeval.truth_Ntrack; truth_part++){         
      if(pfeval.truth_id[truth_part]==pfeval.reco_truthMatch_id[reco_part]){
        double mass = 0;
         if(pfeval.truth_pdg[truth_part]==13) mass = 105.7;
         if(pfeval.truth_pdg[truth_part]==211) mass = 138;
         if(pfeval.truth_pdg[truth_part]==2212 || pfeval.truth_pdg[truth_part]==2112) mass = 938;
           particle_info.reco_truthMatch_mother = pfeval.truth_mother[truth_part];
           temp_truth_mother_id=pfeval.truth_mother[truth_part];
           particle_info.reco_truthMatch_energy = pfeval.truth_startMomentum[truth_part][3]*1000-mass;
           break;
      }
    }
    particle_info.reco_truthMatch_pdg = pfeval.reco_truthMatch_pdg[reco_part];
    particle_info.reco_truthMatch_id = pfeval.reco_truthMatch_id[reco_part];
    particle_info.true_is_n_induced=0;
    for(int truth_mother_part=0; truth_mother_part<pfeval.truth_Ntrack; truth_mother_part++){
      if(pfeval.truth_id[truth_mother_part]!=temp_truth_mother_id) continue;
      if(pfeval.truth_pdg[truth_mother_part]==2112) particle_info.true_is_n_induced = 1;
    }

    // Add some daughter information
    particle_info.flag_has_daught=0;
    particle_info.flag_has_daught_p=0;
    particle_info.flag_has_daught_el=0;
    particle_info.flag_has_daught_pi=0;
    for(int reco_daught_part=0; reco_daught_part<pfeval.reco_Ntrack; reco_daught_part++){
      if(pfeval.reco_mother[reco_daught_part]!=pfeval.reco_id[reco_part]) continue;
      particle_info.flag_has_daught+=1;
      if(pfeval.reco_pdg[reco_daught_part]==2212) particle_info.flag_has_daught_p+=1;
      if(pfeval.reco_pdg[reco_daught_part]==11) particle_info.flag_has_daught_el+=1;
      if(pfeval.reco_pdg[reco_daught_part]==211) particle_info.flag_has_daught_pi+=1;
    }
	    
    particle_info.reco_is_n_induced = 0;
    particle_info.reco_is_g_induced = 0;
    for(int reco_mother_part=0; reco_mother_part<pfeval.reco_Ntrack; reco_mother_part++){
      if(pfeval.reco_id[reco_mother_part]!=pfeval.reco_mother[reco_part]) continue;
      if(pfeval.reco_pdg[reco_mother_part]==2112) particle_info.reco_is_n_induced=1;
      else if(pfeval.reco_pdg[reco_mother_part]==22) particle_info.reco_is_g_induced=1;
    }


    // Save more general variables about the particel
    particle_info.dist_to_vtx = sqrt(pow(part_x-pfeval.reco_nuvtxX,2)+pow(part_y-pfeval.reco_nuvtxY,2)+pow(part_z-pfeval.reco_nuvtxZ,2));

    // Momentum and pid
    double mass = 0;
    if(pfeval.reco_pdg[reco_part]==13) mass = 0.1057;
    if(pfeval.reco_pdg[reco_part]==211) mass = 0.138;
    if(pfeval.reco_pdg[reco_part]==2212) mass = 0.938;
    particle_info.reco_momentum_0 = pfeval.reco_startMomentum[reco_part][0];
    particle_info.reco_momentum_1 = pfeval.reco_startMomentum[reco_part][1];
    particle_info.reco_momentum_2 = pfeval.reco_startMomentum[reco_part][2];
    particle_info.reco_momentum_3 = pfeval.reco_startMomentum[reco_part][3]-mass;
    particle_info.reco_pdg = pfeval.reco_pdg[reco_part];

    // Check if its the leading muon
    if(pfeval.reco_pdg[reco_part]==13 && pfeval.reco_startMomentum[reco_part][3]>pfeval.reco_muonMomentum[3]-tolerance && pfeval.reco_startMomentum[reco_part][3]<pfeval.reco_muonMomentum[3]+tolerance) particle_info.flag_prim_mu = 1;
    else if(pfeval.reco_pdg[reco_part]==13) particle_info.flag_prim_mu = 0;
    else particle_info.flag_prim_mu = -1;

    // Scattering angle
    particle_info.cos_theta = pfeval.reco_startMomentum[reco_part][2] / sqrt( pow(pfeval.reco_startMomentum[reco_part][0],2) + pow(pfeval.reco_startMomentum[reco_part][1],2) + pow(pfeval.reco_startMomentum[reco_part][2],2) );

    // The oriximity to all other particles based off endpoints
    particle_info.proximity=99999999;
    for(int other_reco_part=0; other_reco_part<pfeval.reco_Ntrack; other_reco_part++){
      if (pfeval.reco_id[other_reco_part] == pfeval.reco_id[reco_part]) continue;
      if (pfeval.reco_pdg[other_reco_part] == 2112 || pfeval.reco_pdg[other_reco_part] == 22) continue;
      double temp_proximity_1 =  sqrt(pow(part_x-pfeval.reco_startXYZT[other_reco_part][0],2)+pow(part_y-pfeval.reco_startXYZT[other_reco_part][1],2)+pow(part_z-pfeval.reco_startXYZT[other_reco_part][2],2));
      double temp_proximity_2 = sqrt(pow(part_x-pfeval.reco_endXYZT[other_reco_part][0],2)+pow(part_y-pfeval.reco_endXYZT[other_reco_part][1],2)+pow(part_z-pfeval.reco_endXYZT[other_reco_part][2],2));
      double temp_proximity_3 = sqrt(pow(part_end_x-pfeval.reco_startXYZT[other_reco_part][0],2)+pow(part_end_y-pfeval.reco_startXYZT[other_reco_part][1],2)+pow(part_end_z-pfeval.reco_startXYZT[other_reco_part][2],2));
      double temp_proximity_4 = sqrt(pow(part_end_x-pfeval.reco_endXYZT[other_reco_part][0],2)+pow(part_end_y-pfeval.reco_endXYZT[other_reco_part][1],2)+pow(part_end_z-pfeval.reco_endXYZT[other_reco_part][2],2));
      double temp_proximity = min({temp_proximity_1,temp_proximity_2,temp_proximity_3,temp_proximity_4});
      if(temp_proximity<particle_info.proximity && temp_proximity<99999) particle_info.proximity=temp_proximity;
    }

    // Track lengths and ratio
    particle_info.direct_track_len = sqrt(pow(part_x-part_end_x,2)+pow(part_y-part_end_y,2)+pow(part_z-part_end_z,2));
    particle_info.track_len_ratio = particle_info.direct_track_len/particle_info.track_len;

    // Check the containment
    particle_info.flag_is_contained = 1;
    if(part_end_x < 3 || part_end_x > 250) particle_info.flag_is_contained = 0;
    else if(part_end_y < -113 || part_end_y > 113) particle_info.flag_is_contained = 0;
    else if(part_end_z < 3 || part_end_z > 1035) particle_info.flag_is_contained = 0;

    // Save the larpid vars
    particle_info.reco_larpid_pdg = pfeval.reco_larpid_pdg[reco_part];
    particle_info.reco_larpid_pidScore_el = pfeval.reco_larpid_pidScore_el[reco_part];
    particle_info.reco_larpid_pidScore_ph = pfeval.reco_larpid_pidScore_ph[reco_part];
    particle_info.reco_larpid_pidScore_mu = pfeval.reco_larpid_pidScore_mu[reco_part];
    particle_info.reco_larpid_pidScore_pr = pfeval.reco_larpid_pidScore_pr[reco_part];
    particle_info.reco_larpid_pidScore_pi = pfeval.reco_larpid_pidScore_pi[reco_part];
    particle_info.reco_larpid_proccess = pfeval.reco_larpid_proccess[reco_part];


    // Save the individual spacepoints for easier use in the BDT
    particle_info.spacepoints_q_0 = (particle_info.spacepoints_q->at(0)+10000)*10;
    particle_info.spacepoints_q_bck_0 = (particle_info.spacepoints_q->at(n_spacepoints-1-0)+10000)*10;
    if(n_spacepoints>1){
      particle_info.spacepoints_q_1 = (particle_info.spacepoints_q->at(1)+10000)*10;
      particle_info.spacepoints_q_bck_1 = (particle_info.spacepoints_q->at(n_spacepoints-1-1)+10000)*10;
    }else{
      particle_info.spacepoints_q_1 = -999;
      particle_info.spacepoints_q_bck_1 = -999;
    }
    if(n_spacepoints>2){
      particle_info.spacepoints_q_2 = (particle_info.spacepoints_q->at(2)+10000)*10;
      particle_info.spacepoints_q_bck_2 = (particle_info.spacepoints_q->at(n_spacepoints-1-2)+10000)*10;
    }else{
      particle_info.spacepoints_q_2 = -999;
      particle_info.spacepoints_q_bck_2 = -999;
    }
    if(n_spacepoints>3){
      particle_info.spacepoints_q_3 = (particle_info.spacepoints_q->at(3)+10000)*10;
      particle_info.spacepoints_q_bck_3 = (particle_info.spacepoints_q->at(n_spacepoints-1-3)+10000)*10;
    }else{
      particle_info.spacepoints_q_3 = -999;
      particle_info.spacepoints_q_bck_3 = -999;
    }
    if(n_spacepoints>4){
      particle_info.spacepoints_q_4 = (particle_info.spacepoints_q->at(4)+10000)*10;
      particle_info.spacepoints_q_bck_4 = (particle_info.spacepoints_q->at(n_spacepoints-1-4)+10000)*10;
    }else{
      particle_info.spacepoints_q_4 = -999;
      particle_info.spacepoints_q_bck_4 = -999;
    }
    if(n_spacepoints>5){
      particle_info.spacepoints_q_5 = (particle_info.spacepoints_q->at(5)+10000)*10;
      particle_info.spacepoints_q_bck_5 = (particle_info.spacepoints_q->at(n_spacepoints-1-5)+10000)*10;
    }else{
      particle_info.spacepoints_q_5 = -999;
      particle_info.spacepoints_q_bck_5 = -999;
    }
    if(n_spacepoints>6){
      particle_info.spacepoints_q_6 = (particle_info.spacepoints_q->at(6)+10000)*10;
      particle_info.spacepoints_q_bck_6 = (particle_info.spacepoints_q->at(n_spacepoints-1-6)+10000)*10;
    }else{
      particle_info.spacepoints_q_6 = -999;
      particle_info.spacepoints_q_bck_6 = -999;
    }
    if(n_spacepoints>7){
      particle_info.spacepoints_q_7 = (particle_info.spacepoints_q->at(7)+10000)*10;
      particle_info.spacepoints_q_bck_7 = (particle_info.spacepoints_q->at(n_spacepoints-1-7)+10000)*10;
    }else{
      particle_info.spacepoints_q_7 = -999;
      particle_info.spacepoints_q_bck_7 = -999;
    }
    if(n_spacepoints>8){
      particle_info.spacepoints_q_8 = (particle_info.spacepoints_q->at(8)+10000)*10;
      particle_info.spacepoints_q_bck_8 = (particle_info.spacepoints_q->at(n_spacepoints-1-8)+10000)*10;
    }else{
      particle_info.spacepoints_q_8 = -999;
      particle_info.spacepoints_q_bck_8 = -999;
    }
    if(n_spacepoints>9){
      particle_info.spacepoints_q_9 = (particle_info.spacepoints_q->at(9)+10000)*10;
      particle_info.spacepoints_q_bck_9 = (particle_info.spacepoints_q->at(n_spacepoints-1-9)+10000)*10;
    }else{
      particle_info.spacepoints_q_9 = -999;
      particle_info.spacepoints_q_bck_9 = -999;
    }
    if(n_spacepoints>10){
      particle_info.spacepoints_q_10 = (particle_info.spacepoints_q->at(10)+10000)*10;
      particle_info.spacepoints_q_bck_10 = (particle_info.spacepoints_q->at(n_spacepoints-1-10)+10000)*10;
    }else{
      particle_info.spacepoints_q_10 = -999;
      particle_info.spacepoints_q_bck_10 = -999;
    }
    if(n_spacepoints>11){
      particle_info.spacepoints_q_11 = (particle_info.spacepoints_q->at(11)+10000)*10;
      particle_info.spacepoints_q_bck_11 = (particle_info.spacepoints_q->at(n_spacepoints-1-11)+10000)*10;
    }else{
      particle_info.spacepoints_q_11 = -999;
      particle_info.spacepoints_q_bck_11 = -999;
    }
    if(n_spacepoints>12){
      particle_info.spacepoints_q_12 = (particle_info.spacepoints_q->at(12)+10000)*10;
      particle_info.spacepoints_q_bck_12 = (particle_info.spacepoints_q->at(n_spacepoints-1-12)+10000)*10;
    }else{
      particle_info.spacepoints_q_12 = -999;
      particle_info.spacepoints_q_bck_12 = -999;
    }
    if(n_spacepoints>13){
      particle_info.spacepoints_q_13 = (particle_info.spacepoints_q->at(13)+10000)*10;
      particle_info.spacepoints_q_bck_13 = (particle_info.spacepoints_q->at(n_spacepoints-1-13)+10000)*10;
    }else{
      particle_info.spacepoints_q_13 = -999;
      particle_info.spacepoints_q_bck_13 = -999;
    }
    if(n_spacepoints>14){
      particle_info.spacepoints_q_14 = (particle_info.spacepoints_q->at(14)+10000)*10;
      particle_info.spacepoints_q_bck_14 = (particle_info.spacepoints_q->at(n_spacepoints-1-14)+10000)*10;
    }else{
      particle_info.spacepoints_q_14 = -999;
      particle_info.spacepoints_q_bck_14 = -999;
    }
    if(n_spacepoints>15){
      particle_info.spacepoints_q_15 = (particle_info.spacepoints_q->at(15)+10000)*10;
      particle_info.spacepoints_q_bck_15 = (particle_info.spacepoints_q->at(n_spacepoints-1-15)+10000)*10;
    }else{
      particle_info.spacepoints_q_15 = -999;
      particle_info.spacepoints_q_bck_15 = -999;
    }
    if(n_spacepoints>16){
      particle_info.spacepoints_q_16 = (particle_info.spacepoints_q->at(16)+10000)*10;
      particle_info.spacepoints_q_bck_16 = (particle_info.spacepoints_q->at(n_spacepoints-1-16)+10000)*10;
    }else{
      particle_info.spacepoints_q_16 = -999;
      particle_info.spacepoints_q_bck_16 = -999;
    }
    if(n_spacepoints>17){
      particle_info.spacepoints_q_17 = (particle_info.spacepoints_q->at(17)+10000)*10;
      particle_info.spacepoints_q_bck_17 = (particle_info.spacepoints_q->at(n_spacepoints-1-17)+10000)*10;
    }else{
      particle_info.spacepoints_q_17 = -999;
      particle_info.spacepoints_q_bck_17 = -999;
    }
    if(n_spacepoints>18){
      particle_info.spacepoints_q_18 = (particle_info.spacepoints_q->at(18)+10000)*10;
      particle_info.spacepoints_q_bck_18 = (particle_info.spacepoints_q->at(n_spacepoints-1-18)+10000)*10;
    }else{
      particle_info.spacepoints_q_18 = -999;
      particle_info.spacepoints_q_bck_18 = -999;
    }
    if(n_spacepoints>19){
      particle_info.spacepoints_q_19 = (particle_info.spacepoints_q->at(19)+10000)*10;
      particle_info.spacepoints_q_bck_19 = (particle_info.spacepoints_q->at(n_spacepoints-1-19)+10000)*10;
    }else{
      particle_info.spacepoints_q_19 = -999;
      particle_info.spacepoints_q_bck_19 = -999;
    }
    if(n_spacepoints>20){
      particle_info.spacepoints_q_20 = (particle_info.spacepoints_q->at(20)+10000)*10;
      particle_info.spacepoints_q_bck_20 = (particle_info.spacepoints_q->at(n_spacepoints-1-20)+10000)*10;
    }else{
      particle_info.spacepoints_q_20 = -999;
      particle_info.spacepoints_q_bck_20 = -999;
    }
    if(n_spacepoints>21){
      particle_info.spacepoints_q_21 = (particle_info.spacepoints_q->at(21)+10000)*10;
      particle_info.spacepoints_q_bck_21 = (particle_info.spacepoints_q->at(n_spacepoints-1-21)+10000)*10;
    }else{
      particle_info.spacepoints_q_21 = -999;
      particle_info.spacepoints_q_bck_21 = -999;
    }
    if(n_spacepoints>22){
      particle_info.spacepoints_q_22 = (particle_info.spacepoints_q->at(22)+10000)*10;
      particle_info.spacepoints_q_bck_22 = (particle_info.spacepoints_q->at(n_spacepoints-1-22)+10000)*10;
    }else{
      particle_info.spacepoints_q_22 = -999;
      particle_info.spacepoints_q_bck_22 = -999;
    }
    if(n_spacepoints>23){
      particle_info.spacepoints_q_23 = (particle_info.spacepoints_q->at(23)+10000)*10;
      particle_info.spacepoints_q_bck_23 = (particle_info.spacepoints_q->at(n_spacepoints-1-23)+10000)*10;
    }else{
      particle_info.spacepoints_q_23 = -999;
      particle_info.spacepoints_q_bck_23 = -999;
    }
    if(n_spacepoints>24){
      particle_info.spacepoints_q_24 = (particle_info.spacepoints_q->at(24)+10000)*10;
      particle_info.spacepoints_q_bck_24 = (particle_info.spacepoints_q->at(n_spacepoints-1-24)+10000)*10;
    }else{
      particle_info.spacepoints_q_24 = -999;
      particle_info.spacepoints_q_bck_24 = -999;
    }


  }
  //if(particle_info.reco_pdg>0) print_particle(particle_info);
}

void LEEana::reset_particle(ParticleInfo& particle_info){
    particle_info.spacepoints_x->clear();
    particle_info.spacepoints_y->clear();
    particle_info.spacepoints_z->clear();
    particle_info.spacepoints_q->clear();

    particle_info.spacepoints_q_0=-999;
    particle_info.spacepoints_q_1=-999;
    particle_info.spacepoints_q_2=-999;
    particle_info.spacepoints_q_3=-999;
    particle_info.spacepoints_q_4=-999;
    particle_info.spacepoints_q_5=-999;
    particle_info.spacepoints_q_6=-999;
    particle_info.spacepoints_q_7=-999;
    particle_info.spacepoints_q_8=-999;
    particle_info.spacepoints_q_9=-999;
    particle_info.spacepoints_q_10=-999;
    particle_info.spacepoints_q_11=-999;
    particle_info.spacepoints_q_12=-999;
    particle_info.spacepoints_q_13=-999;
    particle_info.spacepoints_q_14=-999;
    particle_info.spacepoints_q_15=-999;
    particle_info.spacepoints_q_16=-999;
    particle_info.spacepoints_q_17=-999;
    particle_info.spacepoints_q_18=-999;
    particle_info.spacepoints_q_19=-999;
    particle_info.spacepoints_q_20=-999;
    particle_info.spacepoints_q_21=-999;
    particle_info.spacepoints_q_22=-999;
    particle_info.spacepoints_q_23=-999;
    particle_info.spacepoints_q_24=-999;

    particle_info.spacepoints_q_bck_0=-999;
    particle_info.spacepoints_q_bck_1=-999;
    particle_info.spacepoints_q_bck_2=-999;
    particle_info.spacepoints_q_bck_3=-999;
    particle_info.spacepoints_q_bck_4=-999;
    particle_info.spacepoints_q_bck_5=-999;
    particle_info.spacepoints_q_bck_6=-999;
    particle_info.spacepoints_q_bck_7=-999;
    particle_info.spacepoints_q_bck_8=-999;
    particle_info.spacepoints_q_bck_9=-999;
    particle_info.spacepoints_q_bck_10=-999;
    particle_info.spacepoints_q_bck_11=-999;
    particle_info.spacepoints_q_bck_12=-999;
    particle_info.spacepoints_q_bck_13=-999;
    particle_info.spacepoints_q_bck_14=-999;
    particle_info.spacepoints_q_bck_15=-999;
    particle_info.spacepoints_q_bck_16=-999;
    particle_info.spacepoints_q_bck_17=-999;
    particle_info.spacepoints_q_bck_18=-999;
    particle_info.spacepoints_q_bck_19=-999;
    particle_info.spacepoints_q_bck_20=-999;
    particle_info.spacepoints_q_bck_21=-999;
    particle_info.spacepoints_q_bck_22=-999;
    particle_info.spacepoints_q_bck_23=-999;
    particle_info.spacepoints_q_bck_24=-999;

    particle_info.spacepoints_q_med=-999;

    particle_info.flag_prim_mu=-999;

    particle_info.flag_is_contained=-999;

    particle_info.flag_has_daught=-999;
    particle_info.flag_has_daught_p=-999;
    particle_info.flag_has_daught_el=-999;
    particle_info.flag_has_daught_pi=-999;

    particle_info.reco_truthMatch_pdg=-999;
    particle_info.reco_truthMatch_id=-999;
    particle_info.reco_truthMatch_mother=-999;
    particle_info.reco_truthMatch_energy=-999;

    particle_info.reco_momentum_0=-999;
    particle_info.reco_momentum_1=-999;
    particle_info.reco_momentum_2=-999;
    particle_info.reco_momentum_3=-999;
    particle_info.reco_pdg=-999;

    particle_info.reco_larpid_pdg=-999;
    particle_info.reco_larpid_pidScore_el=-999;
    particle_info.reco_larpid_pidScore_ph=-999;
    particle_info.reco_larpid_pidScore_mu=-999;
    particle_info.reco_larpid_pidScore_pr=-999;
    particle_info.reco_larpid_pidScore_pi=-999;
    particle_info.reco_larpid_proccess=-999;

    particle_info.true_is_n_induced=-999;
    particle_info.reco_is_n_induced=-999;
    particle_info.reco_is_g_induced=-999;

    particle_info.dist_to_vtx=-999;
    particle_info.cos_theta=-999;
    particle_info.proximity=-999;

    particle_info.track_len=-999;
    particle_info.direct_track_len=-999;
    particle_info.track_len_ratio=-999;
}

void LEEana::print_particle(ParticleInfo& particle_info){


    std::cout<<"particle_info.spacepoints_q_0 "<<particle_info.spacepoints_q_0<<std::endl;
    std::cout<<"particle_info.spacepoints_q_1 "<<particle_info.spacepoints_q_1<<std::endl;
    std::cout<<"particle_info.spacepoints_q_2 "<<particle_info.spacepoints_q_2<<std::endl;
    std::cout<<"particle_info.spacepoints_q_3 "<<particle_info.spacepoints_q_3<<std::endl;
    std::cout<<"particle_info.spacepoints_q_4 "<<particle_info.spacepoints_q_4<<std::endl;
    std::cout<<"particle_info.spacepoints_q_5 "<<particle_info.spacepoints_q_5<<std::endl;
    std::cout<<"particle_info.spacepoints_q_6 "<<particle_info.spacepoints_q_6<<std::endl;
    std::cout<<"particle_info.spacepoints_q_7 "<<particle_info.spacepoints_q_7<<std::endl;
    std::cout<<"particle_info.spacepoints_q_8 "<<particle_info.spacepoints_q_8<<std::endl;
    std::cout<<"particle_info.spacepoints_q_9 "<<particle_info.spacepoints_q_9<<std::endl;
    std::cout<<"particle_info.spacepoints_q_10 "<<particle_info.spacepoints_q_10<<std::endl;
    std::cout<<"particle_info.spacepoints_q_11 "<<particle_info.spacepoints_q_11<<std::endl;
    std::cout<<"particle_info.spacepoints_q_12 "<<particle_info.spacepoints_q_12<<std::endl;
    std::cout<<"particle_info.spacepoints_q_13 "<<particle_info.spacepoints_q_13<<std::endl;
    std::cout<<"particle_info.spacepoints_q_14 "<<particle_info.spacepoints_q_14<<std::endl;
    std::cout<<"particle_info.spacepoints_q_15 "<<particle_info.spacepoints_q_15<<std::endl;
    std::cout<<"particle_info.spacepoints_q_16 "<<particle_info.spacepoints_q_16<<std::endl;
    std::cout<<"particle_info.spacepoints_q_17 "<<particle_info.spacepoints_q_17<<std::endl;
    std::cout<<"particle_info.spacepoints_q_18 "<<particle_info.spacepoints_q_18<<std::endl;
    std::cout<<"particle_info.spacepoints_q_19 "<<particle_info.spacepoints_q_19<<std::endl;
    std::cout<<"particle_info.spacepoints_q_20 "<<particle_info.spacepoints_q_20<<std::endl;
    std::cout<<"particle_info.spacepoints_q_21 "<<particle_info.spacepoints_q_21<<std::endl;
    std::cout<<"particle_info.spacepoints_q_22 "<<particle_info.spacepoints_q_22<<std::endl;
    std::cout<<"particle_info.spacepoints_q_23 "<<particle_info.spacepoints_q_23<<std::endl;
    std::cout<<"particle_info.spacepoints_q_24 "<<particle_info.spacepoints_q_24<<std::endl;

    std::cout<<"particle_info.spacepoints_q_bck_0 "<<particle_info.spacepoints_q_bck_0<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_1 "<<particle_info.spacepoints_q_bck_1<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_2 "<<particle_info.spacepoints_q_bck_2<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_3 "<<particle_info.spacepoints_q_bck_3<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_4 "<<particle_info.spacepoints_q_bck_4<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_5 "<<particle_info.spacepoints_q_bck_5<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_6 "<<particle_info.spacepoints_q_bck_6<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_7 "<<particle_info.spacepoints_q_bck_7<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_8 "<<particle_info.spacepoints_q_bck_8<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_9 "<<particle_info.spacepoints_q_bck_9<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_10 "<<particle_info.spacepoints_q_bck_10<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_11 "<<particle_info.spacepoints_q_bck_11<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_12 "<<particle_info.spacepoints_q_bck_12<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_13 "<<particle_info.spacepoints_q_bck_13<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_14 "<<particle_info.spacepoints_q_bck_14<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_15 "<<particle_info.spacepoints_q_bck_15<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_16 "<<particle_info.spacepoints_q_bck_16<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_17 "<<particle_info.spacepoints_q_bck_17<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_18 "<<particle_info.spacepoints_q_bck_18<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_19 "<<particle_info.spacepoints_q_bck_19<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_20 "<<particle_info.spacepoints_q_bck_20<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_21 "<<particle_info.spacepoints_q_bck_21<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_22 "<<particle_info.spacepoints_q_bck_22<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_23 "<<particle_info.spacepoints_q_bck_23<<std::endl;
    std::cout<<"particle_info.spacepoints_q_bck_24 "<<particle_info.spacepoints_q_bck_24<<std::endl;

    std::cout<<"particle_info.spacepoints_q_med "<<particle_info.spacepoints_q_med<<std::endl;

    std::cout<<"particle_info.flag_prim_mu "<<particle_info.flag_prim_mu<<std::endl;

    std::cout<<"particle_info.flag_is_contained "<<particle_info.flag_is_contained<<std::endl;

    std::cout<<"particle_info.flag_has_daught "<<particle_info.flag_has_daught<<std::endl;
    std::cout<<"particle_info.flag_has_daught_p "<<particle_info.flag_has_daught_p<<std::endl;
    std::cout<<"particle_info.flag_has_daught_el "<<particle_info.flag_has_daught_el<<std::endl;
    std::cout<<"particle_info.flag_has_daught_pi "<<particle_info.flag_has_daught_pi<<std::endl;

    std::cout<<"particle_info.reco_truthMatch_pdg "<<particle_info.reco_truthMatch_pdg<<std::endl;
    std::cout<<"particle_info.reco_truthMatch_id "<<particle_info.reco_truthMatch_id<<std::endl;
    std::cout<<"particle_info.reco_truthMatch_mother "<<particle_info.reco_truthMatch_mother<<std::endl;
    std::cout<<"particle_info.reco_truthMatch_energy "<<particle_info.reco_truthMatch_energy<<std::endl;

    std::cout<<"particle_info.reco_momentum_0 "<<particle_info.reco_momentum_0<<std::endl;
    std::cout<<"particle_info.reco_momentum_1 "<<particle_info.reco_momentum_1<<std::endl;
    std::cout<<"particle_info.reco_momentum_2 "<<particle_info.reco_momentum_2<<std::endl;
    std::cout<<"particle_info.reco_momentum_3 "<<particle_info.reco_momentum_3<<std::endl;
    std::cout<<"particle_info.reco_pdg "<<particle_info.reco_pdg<<std::endl;

    std::cout<<"particle_info.reco_larpid_pdg "<<particle_info.reco_larpid_pdg<<std::endl;
    std::cout<<"particle_info.reco_larpid_pidScore_el "<<particle_info.reco_larpid_pidScore_el<<std::endl;
    std::cout<<"particle_info.reco_larpid_pidScore_ph "<<particle_info.reco_larpid_pidScore_ph<<std::endl;
    std::cout<<"particle_info.reco_larpid_pidScore_mu "<<particle_info.reco_larpid_pidScore_mu<<std::endl;
    std::cout<<"particle_info.reco_larpid_pidScore_pr "<<particle_info.reco_larpid_pidScore_pr<<std::endl;
    std::cout<<"particle_info.reco_larpid_pidScore_pi "<<particle_info.reco_larpid_pidScore_pi<<std::endl;
    std::cout<<"particle_info.reco_larpid_proccess "<<particle_info.reco_larpid_proccess<<std::endl;

    std::cout<<"particle_info.true_is_n_induced "<<particle_info.true_is_n_induced<<std::endl;
    std::cout<<"particle_info.reco_is_n_induced "<<particle_info.reco_is_n_induced<<std::endl;
    std::cout<<"particle_info.reco_is_g_induced "<<particle_info.reco_is_g_induced<<std::endl;

    std::cout<<"particle_info.dist_to_vtx "<<particle_info.dist_to_vtx<<std::endl;
    std::cout<<"particle_info.cos_theta "<<particle_info.cos_theta<<std::endl;
    std::cout<<"particle_info.proximity "<<particle_info.proximity<<std::endl;

    std::cout<<"particle_info.track_len "<<particle_info.track_len<<std::endl;
    std::cout<<"particle_info.direct_track_len "<<particle_info.direct_track_len<<std::endl;
    std::cout<<"particle_info.track_len_ratio "<<particle_info.track_len_ratio<<std::endl;
}

#endif
