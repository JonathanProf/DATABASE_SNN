#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include "functions.h"

using namespace std;

#define PRINT_ENABLE_INDEX true
#define SAVE_FILE true

#define PATH_SAMPLES_POISSON "../DATABASE_SNN/inputSamples_64ms/%05d_inputSpikesPoisson_64ms.dat"

#define PATH_RESULTS_NET "../DATABASE_SNN/classification/"

#define fileNameIndexNeurons "../DATABASE_SNN/classification/index_Sample_Cpp.dat"

#define TOTAL_SAMPLES static_cast<int>(1000)


int main(int argc, char *argv[])
{
    assert(argc > 2);
    // constants
    const uint16_t NUM_NEURONS = stoi( argv[1] );
    const uint16_t NUM_PIXELS = 784;
    const uint16_t SINGLE_SAMPLE_TIME = 64;
    const string PATH_PARAMETERS_NET = "../DATABASE_SNN/window64ms/BD" + string( argv[1] ) + "_64ms/";
    const uint16_t WG = stoi(argv[2]);
    const string filenameLabels = string(PATH_RESULTS_NET) + string("labels_CppSerial_QT_") + std::to_string(NUM_NEURONS) +string("N_64ms.csv");
    /*
    float v_rest_e = -65.0;     // [mV]
    float v_reset_e = -60.0;    // [mV]
    float v_thresh_e = -52.0;   // [mV]
    int refrac_e = 5;           // [ms]
    
    float v_rest_i = -60.0;     // [mV]
    float v_reset_i = -45.0;    // [mV]
    float v_thresh_i = -40.0;   // [mV]
    int refrac_i = 2;           // [ms]
    int dt = 1;
    */
    assert( sizeof (uint32_t) == 4 );
    assert( sizeof (uint16_t) == 2 );
    assert( sizeof (float) == 4 );
    
    //! array to save the index of excitatory neurons that generate a spike in the time t
    //uint16_t *indexArray;
    //if( (indexArray = (uint16_t *)calloc(TOTAL_SAMPLES*NUM_NEURONS*SINGLE_SAMPLE_TIME, sizeof(uint16_t))) == NULL )
    //    perror("memory allocation for a");
    
    uint16_t tamVector = NUM_NEURONS / 16; //(sizeof (uint16_t)*8)
    uint16_t tamVectorPixels = NUM_PIXELS / 16; //(sizeof (uint16_t)*8)
    
    //! =====     =====     =====
    //! Variables Inicialization
    //! =====     =====     =====
    
    float weights_Ae_Ai_constant = 22.5;
    float weights_Ai_Ae_constant = -120.0;
    
    float *theta = nullptr;
    float *weightsXeAe = nullptr; // connections Xe -> Ae
    float *weightsAeAi = nullptr; // connections Ae -> Ai
    float *weightsAiAe = nullptr; // connections Ae <- Ai
    uint32_t *input_sample = nullptr;
    
    // Check for spiking neurons
    uint16_t *spikesXePre = nullptr; // Spike occurrences Input
    uint16_t *spikesXePos = nullptr; // Spike occurrences Input
    uint16_t *spikes_Ae_Ai_pre = nullptr;
    uint16_t *spikes_Ae_Ai_pos = nullptr;
    uint16_t *spikes_Ai_Ae_pre = nullptr;
    uint16_t *spikes_Ai_Ae_pos = nullptr;
    
    uint32_t *spike_count = nullptr;
    uint16_t* assignments = nullptr;
    
    unsigned short int digits[10] = {0};
    
    float *vE = nullptr;
    float *vI = nullptr;
    
    int *refrac_countE = nullptr;       // Refractory period counters
    int *refrac_countI = nullptr;       // Refractory period counters
    
    //! [Step 1] Data structure initialization
    
    spikesXePre = new(std::nothrow) uint16_t[tamVectorPixels]{0};
    assert(spikesXePre != nullptr);
    
    spikesXePos = new(std::nothrow) uint16_t[tamVectorPixels]{0};
    assert(spikesXePos != nullptr);
    
    spikes_Ae_Ai_pre = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ae_Ai_pre != nullptr);
    
    spikes_Ae_Ai_pos = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ae_Ai_pos != nullptr);
    
    spikes_Ai_Ae_pre = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ai_Ae_pre != nullptr);
    
    spikes_Ai_Ae_pos = new(std::nothrow) uint16_t[tamVector]{0};
    assert(spikes_Ai_Ae_pos != nullptr);
    
    vE = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vE != nullptr );
    
    vI = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vI != nullptr );
    
    for (int indx = 0; indx < NUM_NEURONS; ++indx) {
        vE[indx] = -65.0f;      // v_rest_e = -65.0[mV]
        vI[indx] = -60.0f;    // v_rest_i = -60.0[mV]
    }
    
    refrac_countE = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countE != nullptr );
    
    refrac_countI = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countI != nullptr );
    
    unsigned int tamArrSamples = SINGLE_SAMPLE_TIME/32;
    input_sample = new(std::nothrow) uint32_t[NUM_PIXELS * tamArrSamples]{0};
    assert( input_sample != nullptr );
    
    spike_count = new(std::nothrow) uint32_t[NUM_NEURONS]{0};
    assert( spike_count != nullptr );
    
    //! [Step 2] Loading data from files
    
    weightsXeAe = new(std::nothrow) float[NUM_PIXELS*NUM_NEURONS]{0};
    assert( weightsXeAe != nullptr );
    
    std::string filename(PATH_PARAMETERS_NET);
    filename += "XeAe_" + std::to_string(NUM_NEURONS) + "N_" + std::to_string(SINGLE_SAMPLE_TIME) + "ms.dat";
    getWeights(weightsXeAe, filename, NUM_PIXELS, NUM_NEURONS);
    
    // The weight vector for Ae -> Ai layer is initialized.
    weightsAeAi = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAeAi != nullptr );
    
    for (int i = 0; i < NUM_NEURONS; ++i) {
        weightsAeAi[i*NUM_NEURONS+i] = weights_Ae_Ai_constant;
    }
    
    // The weight vector for Ai -> Ae layer is initialized.
    weightsAiAe = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAiAe != nullptr );
    
    for (int i = 0; i < NUM_NEURONS; ++i) {
        for (int j = 0; j < NUM_NEURONS; ++j) {
            weightsAiAe[i*NUM_NEURONS+j] = (i != j) ? weights_Ai_Ae_constant : 0.0;
        }
    }
    
    //! theta values are loaded into the group of excitatory neurons
    theta = new(std::nothrow) float[NUM_NEURONS]{0};
    assert( theta != nullptr );
    
    filename = (PATH_PARAMETERS_NET);
    filename += "theta_" + std::to_string(NUM_NEURONS) + "N_" + std::to_string(SINGLE_SAMPLE_TIME) + "ms.dat";
    getTheta( theta, filename, NUM_NEURONS );
    
    assignments = new(std::nothrow) uint16_t[NUM_NEURONS]{0};
    assert( assignments != nullptr );
    
    filename = PATH_PARAMETERS_NET;
    filename += "assignments_" + std::to_string(NUM_NEURONS) + "N_" + std::to_string(SINGLE_SAMPLE_TIME) + "ms.dat";
    getAssignments( assignments, filename, NUM_NEURONS );
    
    time_t start, end;
    time(&start);
    //! =====     =====     =====
    //! Run Simulation
    //! =====     =====     =====
    
    // Variables to debug the program and verify the synapse
    float *iSyn_Xe_Ae = new float[NUM_NEURONS]{0.0f};
    float *iSyn_Ai_Ae = new float[NUM_NEURONS]{0.0f};
    
    for ( int numSample = 1; numSample <= TOTAL_SAMPLES; ++numSample ) {
        
        cout << "sample num: " << numSample << ";\t";
        
        char buffer[100];
        sprintf( buffer, PATH_SAMPLES_POISSON ,numSample);
        
        std::string filename(buffer);
        
        getInputSample( input_sample, filename, NUM_PIXELS, tamArrSamples);
        
        //! Simulate network activity for SINGLE_SAMPLE_TIME timesteps.
        for (int t = 0; t < SINGLE_SAMPLE_TIME; ++t)
        {

            // Compute the input spikes to feed the Exc layer
            compute_Xe_Ae_spikes( input_sample,
                                  spikesXePre,
                                  t,
                                  NUM_PIXELS
                                  );
            
            // 1. Update E neurons
            update_exc_neurons(NUM_NEURONS,
                               NUM_PIXELS,
                               iSyn_Xe_Ae,
                               spikesXePos,
                               weightsXeAe,
                               iSyn_Ai_Ae,
                               spikes_Ai_Ae_pos,
                               weightsAiAe,
                               vE,
                               refrac_countE,
                               spikes_Ae_Ai_pre,
                               theta
                               );
            
            // Select the winner neuron
            select_winner_neuron( NUM_NEURONS,
                                  spikes_Ae_Ai_pre,
                                  numSample,
                                  t,
                                  //indexArray,
                                  assignments,
                                  SINGLE_SAMPLE_TIME,
                                  digits,
                                  tamVector,
                                  spike_count
                                  );
            
            // 2. Update I neurons
            update_inh_neurons( NUM_NEURONS,
                                vI,
                                spikes_Ae_Ai_pos,
                                weightsAeAi,
                                refrac_countI,
                                spikes_Ai_Ae_pre,
                                tamVector
                                );
            // ************************************************************* //
            // **     RESET ALL THE VECTORS TO COMPUTE ANOTHER SAMPLE     ** //
            // ************************************************************* //

            // The arrays are exchanged for the next iteration of time
            for (int indx = 0; indx < tamVectorPixels; ++indx) {
                spikesXePos[indx] = spikesXePre[indx];
                spikesXePre[indx] = 0;
            }
            
            for (int indx = 0; indx < tamVector; ++indx){
                spikes_Ae_Ai_pos[indx] = spikes_Ae_Ai_pre[indx];
                spikes_Ai_Ae_pos[indx] = spikes_Ai_Ae_pre[indx];
                spikes_Ae_Ai_pre[indx] = 0;
                spikes_Ai_Ae_pre[indx] = 0;
            }

            for (int i = 0; i < NUM_NEURONS; ++i)
                iSyn_Xe_Ae[i] = 0.0f;

            for (int i = 0; i < NUM_NEURONS; ++i)
                iSyn_Ai_Ae[i] = 0.0f;
            
        } // end of time loop
        
        // classification
        float rates[10] = {0.0f};
        
        //! Count the number of neurons with this label assignment.
        float n_assigns[10] = {0};
        
        for (int indx = 0; indx < NUM_NEURONS; ++indx) {
            ++n_assigns[assignments[indx]];
            
            rates[assignments[indx]] += spike_count[indx];
        }
        
        for (int indx = 0; indx < 10; ++indx) {
            rates[indx] /= n_assigns[indx];
        }
        
        int indWinner = 0;
        float ratesWin = 0;
        
        for (int indx = 0; indx < 10; ++indx) {
            if ( rates[indx] > ratesWin ) {
                indWinner = indx;
                ratesWin = rates[indx];
            }
        }
        
#if PRINT_ENABLE_INDEX == true
        std::cout << "Digit class: " << indWinner << std::endl;
#endif
        
#if SAVE_FILE == true
        ofstream fileLabels;
        fileLabels.open(filenameLabels, std::ofstream::out | std::ofstream::app);
        if (!fileLabels.is_open())
        {
            std::cout << "Error opening file" << __LINE__ << std::endl;
            exit(1);
        }
        
        fileLabels << indWinner << std::endl;
        fileLabels.close();
#endif
        
        //! =====     =====     =====
        //! Reset Variables
        //! =====     =====     =====
        for (int i = 0; i < NUM_NEURONS; ++i) {
            
            vE[i] = -65.0f; // v_rest_e = -65.0 [mV]
            vI[i] = -60.0f; // v_rest_i = -60.0 [mV]
            
            refrac_countE[i] = 0;
            refrac_countI[i] = 0;
            
            spike_count[i] = 0;
        }
        
        for (uint16_t i = 0; i < tamVector; ++i) {
            spikes_Ae_Ai_pos[i] = 0;
            spikes_Ae_Ai_pre[i] = 0;
            spikes_Ai_Ae_pos[i] = 0;
            spikes_Ai_Ae_pre[i] = 0;
        }
        
        for (int i = 0; i < tamVectorPixels; ++i) {
            spikesXePre[i] = 0;
            spikesXePos[i] = 0;
        }
    }
    
    // Recording end time.
    time(&end);
    
    // Calculating total time taken by the program.
    double time_taken = double(end - start);
    cout << "Time taken by program is : " << time_taken << " sec" << endl;
    
//#if PRINT_ENABLE_INDEX == true
    //std::ofstream raw (fileNameIndexNeurons, std::ofstream::binary);
    //assert( raw.is_open() == true );
    //raw.write((char *)indexArray, TOTAL_SAMPLES*NUM_NEURONS*SINGLE_SAMPLE_TIME*sizeof(uint16_t));
    //raw.close();
//#endif
    
    //free(indexArray);
    delete [] theta;
    delete [] weightsXeAe;
    delete [] weightsAeAi;
    delete [] weightsAiAe;
    delete [] input_sample;
    delete [] spikesXePre;
    delete [] spikesXePos;
    delete [] spikes_Ae_Ai_pre;
    delete [] spikes_Ae_Ai_pos;
    delete [] spikes_Ai_Ae_pre;
    delete [] spikes_Ai_Ae_pos;
    delete [] spike_count;
    delete [] assignments;
    delete [] vE;
    delete [] vI;
    delete [] refrac_countE;
    delete [] refrac_countI;
    
    return 0;
}

/*
#define PRINT_EXC_NEURONS_FILE false
#define PRINT_INH_NEURONS_FILE false
#define SAVE_SYNAPSE false

//! Save the synapse in the current directory
#if SAVE_SYNAPSE == true
std::ofstream raw1 ("C:/Users/jonfe/Documents/GitHub/DATABASE_SNN/classification/iSynXeAeCpp.dat", std::ofstream::binary);
assert( raw1.is_open() == true );
raw1.write((char *)iSyn_Xe_Ae, NUM_NEURONS*sizeof(float));
raw1.close();

std::ofstream raw2 ("C:/Users/jonfe/Documents/GitHub/DATABASE_SNN/classification/iSynAiAeCpp.dat", std::ofstream::binary);
assert( raw2.is_open() == true );
raw2.write((char *)iSyn_Ai_Ae, NUM_NEURONS*sizeof(float));
raw2.close();
#endif

// PRINT_EXC_NEURONS_FILE
#if PRINT_EXC_NEURONS_FILE == true
std::ofstream raw3 ("C:/Users/jonfe/Documents/GitHub/DATABASE_SNN/classification/vECpp.dat", std::ofstream::binary);
assert( raw3.is_open() == true );
raw3.write((char *)vE, NUM_NEURONS*sizeof(float));
raw3.close();
#endif

#if PRINT_INH_NEURONS_FILE
            std::ofstream raw ("C:/Users/jonfe/Documents/GitHub/DATABASE_SNN/classification/vICpp.dat", std::ofstream::binary);
            assert( raw.is_open() == true );
            raw.write((char *)vI, NUM_NEURONS*sizeof(float));
            raw.close();
#endif
*/
