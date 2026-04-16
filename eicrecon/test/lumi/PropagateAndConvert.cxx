#include "HepMC3/GenEvent.h"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/Print.h"
#include <iostream>

#include "TF1.h"

#include "../constants.h"

using namespace HepMC3;
using namespace std;

struct positions {
  // SJDK 18/08/23 - These were the original positions before the new design was implemented
  //double ConvStart   = -55609;
  //double ConvEnd     = -55610;
  //double SweeperEnd  = -36390;
  // SJDK 18/08/23 - Updated positions
  //double ConvStart   = -64499.5;
  //double ConvEnd     = -64500.5;
  //double SweeperEnd  = -63270;
  double ConvMiddle     = -58000;
  double AnalyzerStart  = -59400;
};

positions POS;

void PropagateAndConvert(string infile="", string outfile="converterElectrons.hepmc", double Zprop = POS.AnalyzerStart) {

  if( infile.empty() ) {
    cout << "infile argument is blank." << endl;
    return;
  }

  // photon splitting fuction from PDG Eq 34.31
  TF1 *PDGsplitting = new TF1("PDGsplitting","1 - 4/3.*x*(1-x)", 0,1);

  ReaderAscii input_file( infile.data() );
  WriterAscii output_file( outfile.data() );

  int events_parsed = 0;

  while( ! input_file.failed() ) {

    GenEvent evt(Units::GEV,Units::MM);
    // Read event from input file
    input_file.read_event(evt);
    // If reading failed - exit loop
    if( input_file.failed() ) break;

    // Grab Primary Vertex
    FourVector PV = evt.event_pos();

    // loop over vertices starting from second one, first should always be the ep primary vertex
    for( int vtx = 1; vtx < evt.vertices().size(); vtx++ ) {

      std::vector<GenVertexPtr> vertices = evt.vertices();

      GenParticlePtr photonIN = vertices[vtx]->particles_in()[0];
      if( photonIN->pid() != 22 ) {
        std::cout<<"Incoming particle is not a photon!!"<<std::endl;
        continue;
      }
      double E_electron = photonIN->momentum().e() * PDGsplitting->GetRandom();
      double E_positron = photonIN->momentum().e() - E_electron;

      // Remove very low energy electrons
      if( E_electron < 0.001 || E_positron < 0.001 ) { continue; }

      double p_electron = sqrt( pow(E_electron,2) - pow(constants::mass_electron,2) );
      double p_positron = sqrt( pow(E_positron,2) - pow(constants::mass_electron,2) );
      double theta = photonIN->momentum().theta();
      double phi = photonIN->momentum().phi();

      GenParticlePtr electron = std::make_shared<GenParticle>(
          FourVector(
            p_electron*sin(theta)*cos(phi),
            p_electron*sin(theta)*sin(phi),
            p_electron*cos(theta),
            E_electron), +11, 1);
      GenParticlePtr positron = std::make_shared<GenParticle>(
          FourVector(
            p_positron*sin(theta)*cos(phi),
            p_positron*sin(theta)*sin(phi),
            p_positron*cos(theta),
            E_positron), -11, 1);
      GenParticlePtr photonOUT = std::make_shared<GenParticle>(
          photonIN->momentum(), 22, 4);


      vertices[vtx]->add_particle_out( electron );
      vertices[vtx]->add_particle_out( positron );
      vertices[vtx]->add_particle_out( photonOUT );

      // linearly propagate particles to new Z location
      double deltaZ = Zprop - PV.z();
      double deltaX = photonIN->momentum().x() / photonIN->momentum().z() * deltaZ;
      double deltaY = photonIN->momentum().y() / photonIN->momentum().z() * deltaZ;
      FourVector RelVtxAtNewLocation( deltaX, deltaY, deltaZ, 0 );
      // set new vertex (absolute)
      vertices[vtx]->set_position( RelVtxAtNewLocation + PV );

      // remove the old outgoing photon (now it's converted)
      evt.remove_particle( vertices[vtx]->particles_out()[0] );

    }

    // Save event to output file
    output_file.write_event(evt);

    ++events_parsed;
    if( events_parsed%1000 == 0 ) std::cout<<"Events parsed: "<<events_parsed<<std::endl;
  }

  input_file.close();
  output_file.close();

  gApplication->Terminate();
}
