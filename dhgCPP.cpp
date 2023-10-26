#include <iostream>
#include <fstream>
#include <httplib.h>
//#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DHGConstructor {
private:
    std::string BASE_URL;
    httplib::Headers HEADERS;
    std::string dhg_name;
    json dihypergraph;

public:
    DHGConstructor(const std::string &baseUrl, const httplib::Headers &headers, const std::string &name)
        : BASE_URL(baseUrl), HEADERS(headers), dhg_name(name) {
        dihypergraph = {
            {"nodes", json::array()},
            {"hyperedges", json::array()}
        };
    }

    void run_loop() {
        httplib::Client cli(BASE_URL.c_str());
        auto res = cli.Get("/issues", HEADERS);
        if (res && res->status == 200) {
            json issues = json::parse(res->body);
            for (const auto &issue : issues) {
                dihypergraph["nodes"].push_back({
                    {"id", "issue" + std::to_string(issue["id"].get<int>())},
                    {"type", "issue"},
                    {"data", issue}
                });
                dihypergraph["hyperedges"].push_back({
                    {"source", {"user_" + std::to_string(issue["user"]["id"].get<int>())}},
                    {"target", {"issue_" + std::to_string(issue["id"].get<int>())}},
                    {"type", "created_by"},
                    {"data", json::object()}
                });
            }
        }

        // fetch commits and PRs similarly...
        std::ofstream outFile("dihypergraph_" + dhg_name + ".json");
        outFile << std::setw(4) << dihypergraph << std::endl;
        std::cout << "Dihypergraph constructed to dihypergraph_" << dhg_name << ".json" << std::endl;
    }
};

int main() {
    const std::string BASE_URL = "https://api.github.com/repos/plurigrid/ontology";
    httplib::Headers HEADERS = {
        {"Authorization", "token ..."},
        {"Accept", "application/vnd.github.v3+json"}
    };

    DHGConstructor my_dhg(BASE_URL, HEADERS, "my_sample");
    my_dhg.run_loop();

    return 0;
}
