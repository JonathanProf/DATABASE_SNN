#include "functions.h"

void compute_Xe_Ae_spikes( const uint32_t  *input_sample,
                           uint16_t *spikesXePre,
                           const int t,
                           const uint16_t NUM_PIXELS
                           ){

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
}

void update_exc_neurons( const uint16_t NUM_NEURONS,
                         const uint16_t NUM_PIXELS,
                         float *iSyn_Xe_Ae,
                         const uint16_t *spikesXePos,
                         const float *weightsXeAe,
                         float *iSyn_Ai_Ae,
                         const uint16_t *spikes_Ai_Ae_pos,
                         const float *weightsAiAe,
                         float *vE,
                         int *refrac_countE,
                         uint16_t *spikes_Ae_Ai_pre,
                         const float *theta
                         ){

    uint16_t index = 0, group = 0;
    uint16_t indexExc = 0, groupExc = 0;
    float iSyn = 0.0;

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
}

void update_inh_neurons( const uint16_t NUM_NEURONS,
                         float *vI,
                         const uint16_t *spikes_Ae_Ai_pos,
                         const float *weightsAeAi,
                         int *refrac_countI,
                         uint16_t *spikes_Ai_Ae_pre,
                         const uint16_t tamVector
                         ){
    uint16_t index = 0, group = 0;
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
}

void select_winner_neuron( const uint16_t NUM_NEURONS,
                           uint16_t *spikes_Ae_Ai_pre,
                           const int numSample,
                           const int t,
                           //uint16_t *indexArray,
                           const uint16_t* assignments,
                           const uint16_t SINGLE_SAMPLE_TIME,
                           unsigned short int* digits,
                           const uint16_t tamVector,
                           uint32_t *spike_count
                           ){

    uint16_t index = 0, group = 0;
    uint16_t indexExc = 0, groupExc = 0;
    int indexWin = -1;
    for (int indx = 0; indx < NUM_NEURONS; ++indx) {

        index = indx % 16; //(sizeof (uint16_t) * 8)
        group = (indx - index) / 16; //(sizeof (uint16_t) * 8)

        assert( index < 16);
        assert( group < NUM_NEURONS/16);

        if( ( spikes_Ae_Ai_pre[group] & (1 << index) ) != 0 ){
            indexWin = (indexWin == -1) ? indx : indexWin;

            //uint32_t indxArrBD = (numSample-1)*(NUM_NEURONS*SINGLE_SAMPLE_TIME)+indx*SINGLE_SAMPLE_TIME+t;

            //indexArray[indxArrBD] = 1;

            //printf("\nt=[%d]; index neuron[%d]; assignment [%d]\n",t,indx, assignments[indx]);
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
}

void getWeights(float *weights, const std::string fileName, const uint32_t pixel, const uint32_t neuron )
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    float n;
    for (uint32_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < pixel * neuron );
        weights[pos] = n;
    }

    raw.close();
}

void getTheta( float *theta, const std::string fileName, const uint32_t numNeurons )
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    float n;
    for (uint32_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < numNeurons );
        theta[pos] = n;
    }

    raw.close();
}

void getInputSample( uint32_t *input, const std::string fileName, const int row, const int col )
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    uint32_t n;
    for (uint16_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < row * col );
        input[pos] = n;

#if DEBUG_INPUT == 1
        std::cout << std::dec << pos;
        std::cout << " -> " << std::hex << input[pos] << std::endl;
#endif
    }

    raw.close();

}

void getAssignments(uint16_t *assignments , const std::string fileName, const uint32_t numNeurons)
{
    std::fstream raw( fileName, std::ios::in | std::ios::binary );

    assert( raw.is_open() == true );
    uint8_t n;
    for (uint32_t pos = 0; ; ++pos) {
        raw.read(reinterpret_cast<char*>(&n), sizeof(n));
        if( raw.eof() ){
            break;
        }
        assert( pos < numNeurons );
        assignments[pos] = n;
    }

    raw.close();
}
