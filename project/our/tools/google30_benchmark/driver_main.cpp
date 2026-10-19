int main(int argc, char *argv[]) {
    int queries=1;
    ArgMapping amap;
    amap.arg("r",party,"Party 1 server, 2 client");
    amap.arg("addr",address,"Server address");
    amap.arg("port",port,"Server port");
    amap.arg("nt",num_threads,"Threads");
    amap.arg("inputDir",inputDir,"Client CSV input directory");
    amap.arg("q",queries,"Sequential fresh-material queries in one initialized session");
    amap.parse(argc,argv);
    if ((party!=1 && party!=2) || queries<1) return 2;
    google30_measure::begin(party);
    std::vector<MYINT> features;
    int label=-1;
    if(party==2) {
        std::ifstream xf(inputDir+"X.csv"),yf(inputDir+"Y.csv");
        std::string xline,yline;
        if(!std::getline(xf,xline) || !std::getline(yf,yline)) return 3;
        auto values=getFeatures(xline); features.resize(values.size());
        populateFixedVector(features.data(),values,scaleForX);
        label=std::stoi(getLabel(yline)[0]);
    }
    google30_measure::phase("initialization_base_ot");
    initialize();
    for(int query=0;query<queries;++query) {
        google30_measure::query_begin();
        int64_t prediction=-1;
        sirnnFixed(party==2?features.data():nullptr,&prediction);
        google30_measure::query_end();
        if(party==2) std::cout << "Predicted Label:" << prediction << ", Actual Label:" << label << std::endl;
    }
    google30_measure::phase("teardown");
    finalize();
    google30_measure::finish();
    return 0;
}
