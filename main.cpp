#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include "functions.h"

using namespace std;

#define DEBUG 0

#define PRINT_EXC_NEURONS_FILE false
#define PRINT_INH_NEURONS_FILE false
#define PRINT_ENABLE_INDEX false
#define SAVE_FILE true
#define SAVE_SYNAPSE false

#define NUM_PIXELS 784

#define SINGLE_SAMPLE_TIME 64

#define PATH_SAMPLES_POISSON "../DATABASE_SNN/inputSamples_64ms/%05d_inputSpikesPoisson_64ms.dat"

#define PATH_RESULTS_NET "../DATABASE_SNN/classification/"

#define TOTAL_SAMPLES static_cast<int>(10000)

#define fileNameIndexNeurons "../DATABASE_SNN/classification/indexSampleCpp.dat";

int main(int argc, char *argv[])
{
    assert(argc > 2);
    // constants
    const uint16_t NUM_NEURONS = stoi( argv[1] );
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
    uint16_t *indexArray;
    if( (indexArray = (uint16_t *)calloc(TOTAL_SAMPLES*NUM_NEURONS*SINGLE_SAMPLE_TIME, sizeof(uint16_t))) == NULL )
        perror("memory allocation for a");

    uint16_t tamVector = NUM_NEURONS / 16; //(sizeof (uint16_t)*8)
    uint16_t tamVectorPixels = NUM_PIXELS / 16; //(sizeof (uint16_t)*8)

    //! =====     =====     =====
    //! Variables Inicialization
    //! =====     =====     =====
    float iSyn = 0.0;

    float weights_Ae_Ai_constant = 22.5;
    float weights_Ai_Ae_constant = -120.0;

    float *theta = nullptr;
    float *weightsXeAe = nullptr; // connections Xe -> Ae
    float *weightsAeAi = nullptr; // connections Ae -> Ai
    float *weightsAiAe = nullptr; // connections Ae <- Ai
    uint32_t  *input_sample = nullptr;

    // Check for spiking neurons
    uint16_t *spikesXePre = nullptr; // Spike occurrences Input
    uint16_t *spikesXePos = nullptr; // Spike occurrences Input
    uint16_t *spikes_Ae_Ai_pre = nullptr;
    uint16_t *spikes_Ae_Ai_pos = nullptr;
    uint16_t *spikes_Ai_Ae_pre = nullptr;
    uint16_t *spikes_Ai_Ae_pos = nullptr;

    uint32_t *spike_count = nullptr;
    uint8_t *assignments = nullptr;

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

    assignments = new(std::nothrow) uint8_t[NUM_NEURONS]{0};
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

    for (int numSample = 1; numSample <= TOTAL_SAMPLES; ++numSample) {

        cout << numSample << endl;

        if( PRINT_ENABLE_INDEX ){
            cout << "numSample[" << numSample << "]=" << numSample-1 << endl;
        }

        char buffer[100];
        sprintf( buffer, PATH_SAMPLES_POISSON ,numSample);

        std::string filename(buffer);

        getInputSample( input_sample, filename, NUM_PIXELS, tamArrSamples);

        //! Simulate network activity for SINGLE_SAMPLE_TIME timesteps.
        for (int t = 0; t < SINGLE_SAMPLE_TIME; ++t)
        {
            uint32_t datoAnalisis = 0;
            uint32_t resultOP = 0;
            uint8_t desplazamiento = 0;
            uint16_t index = 0, group = 0;

            for (int j = 0; j < NUM_PIXELS; ++j)
            {
                // Los números pares serán el indice del pixel
                // La información del tiempo empieza desde el bit más significativo
                //  0 1
                //  2 3
                //  4 5
                //  6 7
                //  8 9
                // 10 11

                //!* Se desenvuelve el bucle ya que solo son dos posiciones del arreglo
                //input_sample[ j*2   ]
                //input_sample[ j*2+1 ]

                datoAnalisis = ( t < 32 ) ? input_sample[ j*2 ] : input_sample[ j*2+1 ];
                desplazamiento = ( t < 32 ) ? (31 - t) : (63 - t);
                resultOP = datoAnalisis & (1 << desplazamiento);
                resultOP >>= desplazamiento;

                // binary information is verified
                assert( resultOP == 0 or resultOP == 1);

                index = j % 16; //(sizeof (uint16_t) * 8)
                group = (j - index) / 16; //(sizeof (uint16_t) * 8)
                spikesXePre[group] |= (resultOP << index);
            }

            uint16_t indexExc = 0, groupExc = 0;

            for (int i = 0; i < NUM_PIXELS; ++i)
                iSyn_Xe_Ae[i] = 0.0f;

            for (int i = 0; i < NUM_NEURONS; ++i)
                iSyn_Ai_Ae[i] = 0.0f;

            // 1. Update E neurons
            for (unsigned int j = 0; j < NUM_NEURONS; ++j)
            {
                indexExc = j % 16; //(sizeof (uint16_t) * 8)
                groupExc = (j - indexExc) / 16; //(sizeof (uint16_t) * 8)

                assert( indexExc < 16);
                assert( groupExc < NUM_NEURONS/16);

                //! The synapse is updated from the input layer Xe -> Ae
                for (int i = 0; i < NUM_PIXELS; ++i) {

                    index = i % 16; //(sizeof (uint16_t) * 8)
                    group = (i - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( index < 16);
                    assert( group < NUM_PIXELS/16);

                    iSyn_Xe_Ae[j] += ( (spikesXePos[group]) & (1 << index) ) ? weightsXeAe[i*NUM_NEURONS+j] : 0;
                }

                //! synapses are updated from inhibitory neurons Ai -> Ae
                for (unsigned int i = 0; i < NUM_NEURONS; ++i) {

                    index = i % 16; //(sizeof (uint16_t) * 8)
                    group = (i - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( index < 16);
                    assert( group < NUM_NEURONS/16);

                    iSyn_Ai_Ae[j] += ( (spikes_Ai_Ae_pos[group]) & (1 << index) ) ? weightsAiAe[j*NUM_NEURONS+i] : 0.0f;
                }

                iSyn = iSyn_Xe_Ae[j] + iSyn_Ai_Ae[j];

                // Decay voltages and adaptive thresholds.
                vE[j] = 0.9900498390197753906250000 * (vE[j] - (-65.0f)) + (-65.0f);

                if( refrac_countE[j] <= 0 )
                    vE[j] += iSyn;

                // Decrement refractory counters.
                refrac_countE[j] -= 1; // dt = 1

                // Check for spiking neurons.
                spikes_Ae_Ai_pre[groupExc] |= ( (vE[j] > ( (-52.0f) + theta[j])) ? (1 << indexExc) : 0 ) ;

                // Refractoriness, voltage reset, and adaptive thresholds.
                if( (spikes_Ae_Ai_pre[groupExc]) & (1 << indexExc) ){
                    refrac_countE[j] = 5; // refrac_e = 5;      // [ms]
                    vE[j] = -60.0f; // v_reset_e = -60.0;    // [mV]
                }

            }

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

            int indexWin = -1;
            for (int indx = 0; indx < NUM_NEURONS; ++indx) {

                index = indx % 16; //(sizeof (uint16_t) * 8)
                group = (indx - index) / 16; //(sizeof (uint16_t) * 8)

                assert( index < 16);
                assert( group < NUM_NEURONS/16);

                if( ( spikes_Ae_Ai_pre[group] & (1 << index) ) != 0 ){
                    indexWin = (indexWin == -1) ? indx : indexWin;

                    uint32_t indxArrBD = (numSample-1)*(NUM_NEURONS*SINGLE_SAMPLE_TIME)+indx*SINGLE_SAMPLE_TIME+t;

                    indexArray[indxArrBD] = 1;

                    if( PRINT_ENABLE_INDEX ){
                        printf("\nt=[%d]; indexWin[%d]",t,indx); cout << endl;
                    }
                }
            }

            if (indexWin >= 0) {

                for (int index = 0; index < NUM_NEURONS; ++index) {

                    indexExc = index % 16; //(sizeof (uint16_t) * 8)
                    groupExc = (index - indexExc) / 16; //(sizeof (uint16_t) * 8)

                    assert( indexExc < 16);
                    assert( groupExc < NUM_NEURONS/16);

                    if( ( spikes_Ae_Ai_pre[groupExc] & (1 << indexExc) ) != 0){
                        ++digits[ assignments[index] ];
                    }
                }
                for (int indx = 0; indx < tamVector; ++indx) {
                    spikes_Ae_Ai_pre[indx] = 0;
                }
                uint16_t indxWin = indexWin % 16; //(sizeof (uint16_t) * 8)
                uint16_t groupWin = (indexWin - indxWin) / 16; //(sizeof (uint16_t) * 8)

                assert( indxWin < 16);
                assert( groupWin < NUM_NEURONS/16);

                spikes_Ae_Ai_pre[groupWin] |= (1 << indxWin);
                ++spike_count[indexWin];
            }

            // 2. Update I neurons
            uint16_t indexInh = 0, groupInh = 0;
            float iSynInh = 0.0f;
            for (int i = 0; i < NUM_NEURONS; ++i)
            {
                // Decay voltages.
                vI[i] = 0.904837429523468017578125 * (vI[i] - (-60.0f)) + (-60.0f);
                iSynInh = 0.0f;

                // Integrate inputs.
                for (int j = 0; j < NUM_NEURONS; ++j) {
                    index = j % 16; //(sizeof (uint16_t) * 8)
                    group = (j - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( index < 16);
                    assert( group < NUM_NEURONS/16);

                    iSynInh += ( (spikes_Ae_Ai_pos[group]) & (1 << index) ) ? weightsAeAi[i*NUM_NEURONS+j] : 0.0f;
                }

                if( refrac_countI[i] > 0 )
                    iSynInh = 0;

                vI[i] += iSynInh;

                // Decrement refractory counters.
                refrac_countI[i] -= 1;

                // Check for spiking neurons.
                if ( vI[i] > -40.0 )
                {
                    refrac_countI[i] =  2; // refrac_i = 2; // [ms]
                    vI[i] =  -45.0f; // v_reset_i = -45.0;    // [mV]
                    for (int j = 0; j < tamVector; ++j) {
                        spikes_Ai_Ae_pre[j] = 0;
                    }

                    indexInh = i % 16; //(sizeof (uint16_t) * 8)
                    groupInh = (i - index) / 16; //(sizeof (uint16_t) * 8)

                    assert( indexInh < 16);
                    assert( groupInh < NUM_NEURONS/16);

                    spikes_Ai_Ae_pre[groupInh] |= (1 << indexInh);
                }

            }

#if PRINT_INH_NEURONS_FILE
            std::ofstream raw ("C:/Users/jonfe/Documents/GitHub/DATABASE_SNN/classification/vICpp.dat", std::ofstream::binary);
            assert( raw.is_open() == true );
            raw.write((char *)vI, NUM_NEURONS*sizeof(float));
            raw.close();
#endif

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

        }

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

#if PRINT_ENABLE_INDEX == true
    std::ofstream raw (fileNameIndexNeurons, std::ofstream::binary);
    assert( raw.is_open() == true );
    raw.write((char *)indexArray, TOTAL_SAMPLES*NUM_NEURONS*SINGLE_SAMPLE_TIME*sizeof(uint16_t));
    raw.close();
#endif

    free(indexArray);
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
