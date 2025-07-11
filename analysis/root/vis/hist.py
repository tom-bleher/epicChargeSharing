#!/usr/bin/env python3

import subprocess
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_branch_list():
    """Get list of all branches from the ROOT file"""
    print("Getting branch list from ROOT file...")
    
    # Create a simple ROOT script to get branch names
    script_content = '''
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TObjArray.h>
#include <iostream>

void get_branches() {
    TFile* file = TFile::Open("/home/tomble/epicChargeSharingOutput.root", "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Could not open file" << std::endl;
        return;
    }
    
    TTree* tree = (TTree*)file->Get("Hits");
    if (!tree) {
        std::cerr << "Error: Could not find tree" << std::endl;
        file->Close();
        return;
    }
    
    TObjArray* branchList = tree->GetListOfBranches();
    for (int i = 0; i < branchList->GetEntries(); ++i) {
        TBranch* branch = (TBranch*)branchList->At(i);
        std::string branchName = branch->GetName();
        
        // Skip vector branches
        if (branchName.find("GridNeighborhood") == std::string::npos) {
            std::cout << branchName << std::endl;
        }
    }
    
    file->Close();
}
'''
    
    # Write the script
    with open('get_branches.C', 'w') as f:
        f.write(script_content)
    
    # Run ROOT to get branch names
    try:
        result = subprocess.run(['root', '-l', '-b', '-q', 'get_branches.C'], 
                              capture_output=True, text=True, timeout=300)
        branches = [line.strip() for line in result.stdout.split('\n') if line.strip() and 'Processing' not in line]
        
        # Clean up
        if os.path.exists('get_branches.C'):
            os.remove('get_branches.C')
            
        print(f"Found {len(branches)} branches to process")
        return branches
    except Exception as e:
        print(f"Error getting branch list: {e}")
        return []

def process_branch(branch_name):
    """Process a single branch in a separate ROOT process"""
    print(f"Processing branch: {branch_name}")
    
    # Create ROOT script for this branch
    script_content = f'''
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TBranch.h>
#include <TPaveStats.h>
#include <TLatex.h>
#include <TList.h>
#include <TText.h>
#include <TNamed.h>
#include <string>
#include <iostream>

std::string extractUnit(const std::string& title) {{
    size_t start = title.find("[");
    size_t end = title.find("]");
    if (start != std::string::npos && end != std::string::npos && end > start) {{
        return title.substr(start, end - start + 1);
    }}
    return "";
}}

std::string getAxisTitle(const std::string& branchName, const std::string& branchTitle, double pixelPitch = 0.0) {{
    std::string unit = extractUnit(branchTitle);
    std::string cleanName = branchName;
    
    if (cleanName.find("3D") != std::string::npos) {{
        cleanName = cleanName.substr(2);
    }}
    
    std::string formatted = "";
    for (size_t i = 0; i < cleanName.length(); ++i) {{
        if (i > 0 && isupper(cleanName[i]) && islower(cleanName[i-1])) {{
            formatted += " ";
        }}
        formatted += cleanName[i];
    }}
    
    // Add pixel pitch information for PixelX/PixelY
    if ((branchName.find("PixelX") != std::string::npos || branchName.find("PixelY") != std::string::npos) && pixelPitch > 0) {{
        if (!unit.empty()) {{
            return formatted + " " + unit + " (bin size: " + std::to_string(pixelPitch) + " mm)";
        }} else {{
            return formatted + " (bin size: " + std::to_string(pixelPitch) + " mm)";
        }}
    }}
    
    if (!unit.empty()) {{
        return formatted + " " + unit;
    }}
    return formatted;
}}

void process_{branch_name.replace("3D", "ThreeD").replace("Power", "Pwr").replace("Lorentz", "Lor")}() {{
    std::string branchName = "{branch_name}";
    std::string filename = "/home/tomble/epicChargeSharingOutput.root";
         std::string outputDir = "histograms";
    
    TFile* file = TFile::Open(filename.c_str(), "READ");
    if (!file || file->IsZombie()) {{
        std::cerr << "Error: Could not open file" << std::endl;
        return;
    }}
    
    TTree* tree = (TTree*)file->Get("Hits");
    if (!tree) {{
        std::cerr << "Error: Could not find tree" << std::endl;
        file->Close();
        return;
    }}
    
    TBranch* branch = tree->GetBranch(branchName.c_str());
    if (!branch) {{
        std::cerr << "Error: Could not find branch " << branchName << std::endl;
        file->Close();
        return;
    }}
    
    std::string branchTitle = branch->GetTitle();
    
    // Read pixel pitch from metadata for PixelX/PixelY histograms
    double pixelPitch_mm = 0.1; // default value
    TNamed* pixelSizeObj = (TNamed*)file->Get("GridPixelSize_mm");
    if (pixelSizeObj) {{
        pixelPitch_mm = std::stod(pixelSizeObj->GetTitle());
    }}
    
    // Calculate min/max
    Double_t minVal = tree->GetMinimum(branchName.c_str());
    Double_t maxVal = tree->GetMaximum(branchName.c_str());
    
    // Determine histogram parameters
    int nbins = 100;
    double xmin = minVal;
    double xmax = maxVal;
    
    if (branchName.find("PixelX") != std::string::npos || branchName.find("PixelY") != std::string::npos) {{
        // Use pixel pitch as bin size for PixelX and PixelY
        double range = maxVal - minVal;
        double paddedMin = minVal - 0.5 * pixelPitch_mm;
        double paddedMax = maxVal + 0.5 * pixelPitch_mm;
        nbins = (int)std::ceil((paddedMax - paddedMin) / pixelPitch_mm);
        xmin = paddedMin;
        xmax = paddedMin + nbins * pixelPitch_mm; // Ensure exact alignment
        std::cout << "Using pixel pitch " << pixelPitch_mm << " mm for " << branchName << " (bins: " << nbins << ")" << std::endl;
    }} else if (branchName.find("Chi2") != std::string::npos) {{
        xmin = 0;
        xmax = maxVal * 1.1;
        nbins = 50;
    }} else if (branchName.find("Pp") != std::string::npos) {{
        xmin = 0;
        xmax = 1;
        nbins = 50;
    }} else if (branchName.find("DOF") != std::string::npos) {{
        nbins = std::min(50, (int)(maxVal - minVal + 1));
    }} else if (branchName.find("Angle") != std::string::npos) {{
        xmin = 0;
        xmax = 360;
        nbins = 72;
    }} else if (branchName.find("Delta") != std::string::npos || branchName.find("Error") != std::string::npos || branchName.find("Err") != std::string::npos) {{
        double range = maxVal - minVal;
        xmin = minVal - 0.1*range;
        xmax = maxVal + 0.1*range;
        nbins = 80;
    }} else if (branchName.find("IsPixelHit") != std::string::npos || branchName.find("Successful") != std::string::npos) {{
        xmin = -0.5;
        xmax = 1.5;
        nbins = 2;
    }} else {{
        double range = maxVal - minVal;
        if (range > 0) {{
            xmin = minVal - 0.05*range;
            xmax = maxVal + 0.05*range;
        }}
    }}
    
    // Create histogram with unique name
    std::string histName = "h_" + branchName + "_temp";
    TH1D* hist = new TH1D(histName.c_str(), branchTitle.c_str(), nbins, xmin, xmax);
    
    // Fill histogram using TTree::Draw and get the filled histogram from ROOT directory
    std::string drawCommand = branchName + ">>" + histName;
    tree->Draw(drawCommand.c_str(), "", "goff");
    
    // Get the filled histogram from ROOT's directory
    TH1D* filledHist = (TH1D*)gDirectory->Get(histName.c_str());
    if (!filledHist) {{
        std::cerr << "Error: Could not retrieve filled histogram for " << branchName << std::endl;
        delete hist;
        file->Close();
        return;
    }}
    
    std::cout << "Filled histogram for " << branchName << " with " << filledHist->GetEntries() << " entries" << std::endl;
    
    // Set histogram properties on the filled histogram
    filledHist->SetLineColor(kBlue+1);
    filledHist->SetFillColor(kBlue-9);
    filledHist->SetFillStyle(1001);
    filledHist->SetLineWidth(2);
    // Set branch description as main title, with branch name to appear in stats box
    filledHist->SetTitle(branchTitle.c_str());
    filledHist->SetName(branchName.c_str());
    filledHist->GetXaxis()->SetTitle(getAxisTitle(branchName, branchTitle, pixelPitch_mm).c_str());
    filledHist->GetYaxis()->SetTitle("Entries");
    filledHist->GetXaxis()->SetTitleSize(0.045);
    filledHist->GetYaxis()->SetTitleSize(0.045);
    filledHist->GetXaxis()->SetLabelSize(0.04);
    filledHist->GetYaxis()->SetLabelSize(0.04);
    filledHist->GetXaxis()->SetTitleOffset(1.1);
    filledHist->GetYaxis()->SetTitleOffset(1.0);
    
    // Create canvas - sized for standard presentation slide (16:9 aspect ratio)
    TCanvas* canvas = new TCanvas(("c_" + branchName).c_str(), branchTitle.c_str(), 1600, 900);
    // Tighter margins to reduce white space on horizontal edges
    canvas->SetMargin(0.08, 0.03, 0.12, 0.08);
    
    // Set style
    gStyle->SetOptStat(1111);  // Show entries, mean, std dev
    gStyle->SetStatBorderSize(1);
    gStyle->SetStatColor(0);
    gStyle->SetStatFont(42);
    gStyle->SetStatFontSize(0.032);
    gStyle->SetStatX(0.94);
    gStyle->SetStatY(0.94);
    gStyle->SetStatFormat("6.3g");  // Better number formatting
    
    // Draw and save
    filledHist->Draw("HIST");
    canvas->Update();
    
    // Customize stats box to show branch name as first line
    TPaveStats* stats = (TPaveStats*)filledHist->FindObject("stats");
    if (stats) {{
        stats->SetName("mystats");
        TList* listOfLines = stats->GetListOfLines();
        TText* tconst = stats->GetLineWith(filledHist->GetName());
        if (tconst) {{
            listOfLines->Remove(tconst);
        }}
        // Add branch name as first line
        TLatex* myt = new TLatex(0, 0, branchName.c_str());
        myt->SetTextFont(42);
        myt->SetTextSize(0.032);
        listOfLines->AddFirst(myt);
        stats->SetOptStat(1111);
        canvas->Modified();
    }}
    
    std::string svgFilename = outputDir + "/" + branchName + ".svg";
    canvas->SaveAs(svgFilename.c_str());
    
    std::cout << "Completed: " << svgFilename << std::endl;
    
    // Cleanup
    delete hist;
    delete canvas;
    file->Close();
}}
'''
    
    # Write the script
    safe_branch_name = branch_name.replace("3D", "ThreeD").replace("Power", "Pwr").replace("Lorentz", "Lor")
    script_file = f'process_{safe_branch_name}.C'
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    try:
        # Run ROOT
        result = subprocess.run(['root', '-l', '-b', '-q', script_file], 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print(f"Successfully processed: {branch_name}")
            return True
        else:
            print(f"Error processing {branch_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {branch_name}")
        return False
    except Exception as e:
        print(f"Exception processing {branch_name}: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(script_file):
            os.remove(script_file)

def main():
    start_time = time.time()
    
    print("Starting FIXED parallel histogram creation...")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    
    # Create output directory
    os.makedirs("histograms", exist_ok=True)
    
    # Get branch list
    branches = get_branch_list()
    if not branches:
        print("No branches found!")
        return
    
    # Use parallel processing
    max_workers = multiprocessing.cpu_count()
    print(f"Using {max_workers} parallel workers")
    
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_branch = {executor.submit(process_branch, branch): branch for branch in branches}
        
        # Process completed tasks
        for future in as_completed(future_to_branch):
            branch = future_to_branch[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
                print(f"Progress: {successful + failed}/{len(branches)} completed")
            except Exception as e:
                print(f"Exception for {branch}: {e}")
                failed += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nCompleted processing!")
    print(f"Total branches: {len(branches)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time taken: {duration:.1f} seconds")
    
    # Count SVG files
    svg_count = len([f for f in os.listdir("histograms") if f.endswith('.svg')])
    print(f"SVG files created: {svg_count}")
    print("All histograms saved in directory: histograms")

if __name__ == "__main__":
    main() 