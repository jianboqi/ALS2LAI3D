#include<string>
#include <fstream>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/Delaunay_triangulation_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Alpha_shape_vertex_base_3<K>               Vb;
typedef CGAL::Alpha_shape_cell_base_3<K>                 Fb;
typedef CGAL::Triangulation_data_structure_3<Vb, Fb>      Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location>  Delaunay;
typedef CGAL::Alpha_shape_3<Delaunay>                    Alpha_shape_3;
typedef K::Point_3                                       Point;
typedef Alpha_shape_3::Alpha_iterator                    Alpha_iterator;
typedef Alpha_shape_3::NT                                NT;
using namespace std;

typedef unsigned int uint;

extern "C" __declspec(dllexport) double do_ASA(float* arr, int rows, const char* save_path,float alpha_value)
{
	double volume = 0;
	Delaunay dt;
	for (int i = 0; i < rows; i++)
		dt.insert(Point(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]));
	double alpha = 0.1;
	if (alpha_value != -1)
		alpha = alpha_value;
	Alpha_shape_3 as(dt,alpha);
	//Alpha_shape_3 as(dt);

	if (alpha_value == -1)
	{
		Alpha_iterator opt = as.find_optimal_alpha(1);
		as.set_alpha(*opt);
	}

	std::vector<Point> vertices;
	std::vector<uint> faces;

	std::ofstream out(save_path);
	boost::unordered_set< Alpha_shape_3::Cell_handle > marked_cells;
	std::vector< Alpha_shape_3::Cell_handle > queue;
	queue.push_back(as.infinite_cell());

	while (!queue.empty())
	{
		Alpha_shape_3::Cell_handle back = queue.back();
		queue.pop_back();

		if (!marked_cells.insert(back).second) continue; //already visited

		for (int i = 0; i < 4; ++i)
		{
			if (as.classify(Alpha_shape_3::Facet(back, i)) == Alpha_shape_3::EXTERIOR &&
				marked_cells.count(back->neighbor(i)) == 0)
				queue.push_back(back->neighbor(i));
		}
	}
	boost::unordered_map< Alpha_shape_3::Vertex_handle, std::size_t> vids;
	std::vector< Alpha_shape_3::Facet > regular_facets;
	as.get_alpha_shape_facets(std::back_inserter(regular_facets), Alpha_shape_3::REGULAR);

	std::vector<Alpha_shape_3::Facet> filtered_regular_facets;
	for (Alpha_shape_3::Facet f : regular_facets)
	{
		if (marked_cells.count(f.first) == 1)
			filtered_regular_facets.push_back(f);
		else
		{
			f = as.mirror_facet(f);
			if (marked_cells.count(f.first) == 1)
				filtered_regular_facets.push_back(f);
		}
	}

	float min_x = 1e10, min_y = 1e10, min_z = 1e10;
	for (Alpha_shape_3::Facet f : filtered_regular_facets)
	{
		for (int i = 1; i < 4; ++i)
		{
			Alpha_shape_3::Vertex_handle vh = f.first->vertex((f.second + i) % 4);
			if (vids.insert(std::make_pair(vh, vertices.size())).second)
			{
				Point pt= vh->point();
				if (pt.x() < min_x)
					min_x = pt.x();
				if (pt.y() < min_y)
					min_y = pt.y();
				if (pt.z() < min_z)
					min_z = pt.z();
				out << "v " << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
				vertices.push_back(pt);
			}
		}
	}
	Point min_pt(min_x, min_y, min_z);

	for (const Alpha_shape_3::Facet& f : filtered_regular_facets)
	{
		for (int i = 0; i < 3; ++i)
		{
			Alpha_shape_3::Vertex_handle vh = f.first->vertex(as.vertex_triple_index(f.second, i));
			faces.push_back(vids[vh]);
		}
		out << "f " << faces[faces.size()-3] + 1 << " " << faces[faces.size() - 2] + 1 << " " << faces[faces.size() - 1] + 1 << "\n";
		auto pt1 = vertices[faces[faces.size() - 3]] - min_pt;
		auto pt2 = vertices[faces[faces.size() - 2]] - min_pt;
		auto pt3 = vertices[faces[faces.size() - 1]] - min_pt;
		double signedVol = (-(pt3.x()*pt2.y()*pt1.z())
			+ (pt2.x()*pt3.y()*pt1.z())
			+ (pt3.x()*pt1.y()*pt2.z())
			- (pt1.x()*pt3.y()*pt2.z())
			- (pt2.x()*pt1.y()*pt3.z())
			+ (pt1.x()*pt2.y()*pt3.z())) / 6.0;
		volume += signedVol;
	}
	out.close();
	return volume;
}

/*
double do_ASA2(float* arr, int rows, const char* save_path)
{
	double volume = 0;
	Delaunay dt;
	for (int i = 0; i < rows; i++)
		dt.insert(Point(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]));
	Alpha_shape_3 as(dt);
	Alpha_iterator opt = as.find_optimal_alpha(1);
	as.set_alpha(*opt);

	std::vector<Point> vertices;
	std::vector<uint> faces;

	std::ofstream out(save_path);
	boost::unordered_set< Alpha_shape_3::Cell_handle > marked_cells;
	std::vector< Alpha_shape_3::Cell_handle > queue;
	queue.push_back(as.infinite_cell());

	while (!queue.empty())
	{
		Alpha_shape_3::Cell_handle back = queue.back();
		queue.pop_back();

		if (!marked_cells.insert(back).second) continue; //already visited

		for (int i = 0; i < 4; ++i)
		{
			if (as.classify(Alpha_shape_3::Facet(back, i)) == Alpha_shape_3::EXTERIOR &&
				marked_cells.count(back->neighbor(i)) == 0)
				queue.push_back(back->neighbor(i));
		}
	}
	boost::unordered_map< Alpha_shape_3::Vertex_handle, std::size_t> vids;
	std::vector< Alpha_shape_3::Facet > regular_facets;
	as.get_alpha_shape_facets(std::back_inserter(regular_facets), Alpha_shape_3::REGULAR);

	std::vector<Alpha_shape_3::Facet> filtered_regular_facets;
	for (Alpha_shape_3::Facet f : regular_facets)
	{
		if (marked_cells.count(f.first) == 1)
			filtered_regular_facets.push_back(f);
		else
		{
			f = as.mirror_facet(f);
			if (marked_cells.count(f.first) == 1)
				filtered_regular_facets.push_back(f);
		}
	}

	float min_x = 1e10, min_y = 1e10, min_z = 1e10;
	for (Alpha_shape_3::Facet f : filtered_regular_facets)
	{
		for (int i = 1; i < 4; ++i)
		{
			Alpha_shape_3::Vertex_handle vh = f.first->vertex((f.second + i) % 4);
			if (vids.insert(std::make_pair(vh, vertices.size())).second)
			{
				Point pt= vh->point();
				if (pt.x() < min_x)
					min_x = pt.x();
				if (pt.y() < min_y)
					min_y = pt.y();
				if (pt.z() < min_z)
					min_z = pt.z();
				out << "v " << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
				vertices.push_back(pt);
			}
		}
	}
	Point min_pt(min_x, min_y, min_z);

	for (const Alpha_shape_3::Facet& f : filtered_regular_facets)
	{
		for (int i = 0; i < 3; ++i)
		{
			Alpha_shape_3::Vertex_handle vh = f.first->vertex(as.vertex_triple_index(f.second, i));
			faces.push_back(vids[vh]);
		}
		out << "f " << faces[faces.size()-3] + 1 << " " << faces[faces.size() - 2] + 1 << " " << faces[faces.size() - 1] + 1 << "\n";
		auto pt1 = vertices[faces[faces.size() - 3]] - min_pt;
		auto pt2 = vertices[faces[faces.size() - 2]] - min_pt;
		auto pt3 = vertices[faces[faces.size() - 1]] - min_pt;
		double signedVol = (-(pt3.x()*pt2.y()*pt1.z())
			+ (pt2.x()*pt3.y()*pt1.z())
			+ (pt3.x()*pt1.y()*pt2.z())
			- (pt1.x()*pt3.y()*pt2.z())
			- (pt2.x()*pt1.y()*pt3.z())
			+ (pt1.x()*pt2.y()*pt3.z())) / 6.0;
		volume += signedVol;
	}
	out.close();
	return volume;
}

int main()
{
	ifstream in("C:\\Users\\ZXZ\\Desktop\\test.txt");
	vector<float> pts;
	while(!in.eof())
	{
		float x;
		in >> x;
		pts.push_back(x);
	}
	do_ASA2(&pts[0], pts.size() / 3, "bbb.obj");
	return 0;
}*/